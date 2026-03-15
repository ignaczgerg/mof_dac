"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from collections.abc import Callable
from functools import cache, partial
from logging import getLogger
from contextlib import nullcontext
from typing import Annotated, Literal, TypeAlias, assert_never, final, cast, Optional
import os
import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from einops import reduce
from ase.data import atomic_masses
from torch.utils.data import DataLoader, DistributedSampler, get_worker_info, RandomSampler, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.batch import Batch
from typing_extensions import override, TypeVar
from torchmetrics import SumMetric
from dataclasses import dataclass

from jmp.lightning import TypedConfig
from jmp.lightning.util.typed import TypedModuleDict, TypedModuleList

from ...modules import transforms as T
from ...modules.dataset import dataset_transform as DT
from ...modules.dataset.common import CommonDatasetConfig, wrap_common_dataset
from ...modules.dataset.concat_dataset import MTDatasetConfig, MTSampledDataset
from ...modules.transforms.normalize import NormalizationConfig

from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from .base import (
    FinetuneConfigBase, FinetuneModelBase, FinetuneLmdbDatasetConfig, 
    DatasetType, RLPConfig
    )
from .metrics import FinetuneMetrics, MetricPair, MetricsConfig, MTFinetuneMetrics
from jmp.tasks.finetune import model_wrapper_base as models
from jmp.tasks.finetune.model_wrapper_base import (
    ModelWrapperBase, MulticlassClassificationTargetConfig, 
    BinaryClassificationTargetConfig, OutputHeadInput
    )
import torch.distributed as dist
from jmp.modules.atom_batch_sampler import AtomBucketBatchSampler

log = getLogger(__name__)

AdsorptionDbTarget: TypeAlias = Literal[
    'qst_co2', 
    'qst_h2o',  
    'qst_n2',
    'kh_co2', 
    'kh_h2o', 
    'kh_n2', 
    'selectivity_co2_h2o', 
    'selectivity_co2_n2', 
    'co2_uptake', 
    'n2_uptake'
]

def _identity_transform(data: BaseData, *, training: bool) -> BaseData:
    return data

def _collate_exclude(data_list, *, exclude_keys):
    # added this pass always lists.
    # we needed this for inference when we handle sinle cif files.
    if isinstance(data_list, Batch):
        data_list = data_list.to_data_list()
    elif isinstance(data_list, BaseData):
        data_list = [data_list]
    elif not isinstance(data_list, (list, tuple)):
        raise TypeError(f"collate received unexpected type: {type(data_list)}")
    return Batch.from_data_list(list(data_list), exclude_keys=exclude_keys)

def _seed_worker(worker_id: int):
    worker_info = get_worker_info()
    base_seed = torch.initial_seed() % 2**32
    g = torch.Generator()
    g.manual_seed(base_seed + worker_id)
    try:
        np.random.seed(base_seed + worker_id)
        random.seed(base_seed + worker_id)
    except Exception:
        pass

@dataclass(eq=False)
class AdsorptionFilter:
    is_train: bool = False
    max_natoms: Optional[int] = None
    def __call__(self, data: BaseData) -> bool:
        ok = (
            (data.kh_co2  < 1.0) and
            (data.kh_h2o  < 1.0) and
            (data.qst_co2 < 0.0) and
            (data.qst_h2o < 0.0) and
            (data.co2_uptake > 0.0) and # removes nans
            (data.selectivity_co2_h2o > 0.0) # removes nans
        )
        # if self.is_train and (self.max_natoms is not None):
        #     ok = ok and (data.natoms < self.max_natoms)
        if self.max_natoms is not None:
            ok = ok and (data.natoms < self.max_natoms)
        return ok

class _EmptyDataset(Dataset):
    def __len__(self): return 0
    def __getitem__(self, idx): raise IndexError

class TaskConfig(TypedConfig):
    name: str
    """Name of the task."""

    train_dataset: FinetuneLmdbDatasetConfig | None
    """Train dataset configuration."""

    val_dataset: FinetuneLmdbDatasetConfig 
    """Validation dataset configuration."""
    
    test_dataset: FinetuneLmdbDatasetConfig
    """Test dataset configuration."""

    node_energy_reduction: Literal["sum", "mean"] = "sum"
    """How to reduce the node energy scalar contributions (to get the total energy)."""

    additional_units: list[str] = []
    """Additional units to log for this task."""

    graph_scalar_targets: list[str] = []
    """List of graph scalar targets (e.g., energy)"""
    graph_classification_targets: list[BinaryClassificationTargetConfig | MulticlassClassificationTargetConfig] = []
    """List of graph classification targets (e.g., is_metal)"""
    graph_classification_adabin_targets: list[MulticlassClassificationTargetConfig] = []
    """List of graph regression targets reformulated as classification targets (e.g., energy [using adabin])"""
    node_vector_targets: list[str] = []
    """List of node vector targets (e.g., force)"""
    
    graph_scalar_loss_coefficients: dict[str, float] = {}
    """Loss coefficients for graph scalar targets"""
    graph_classification_loss_coefficients: dict[str, float] = {}
    """Loss coefficients for graph classification targets"""
    node_vector_loss_coefficients: dict[str, float] = {}
    """Loss coefficients for node vector targets"""


    normalization: dict[str, NormalizationConfig] | None = None
    """
    Normalization to apply to the target values.
    Each key is the name of the target value
    and the value is a dict with the mean and std.
    """

class MOFFeatureConfig(TypedConfig):
    enabled: bool = False                 # turn PLD/LCD conditioning on/off
    feature_name: str = "pld"             # "pld", "lcd", or other features
    fusion_type: str = "early"            # "early" or "late" fustion
    hidden_dim: int = 128                  # embedding projection size

class AdsorptionDbConfig(FinetuneConfigBase):
    graph_scalar_targets: list[str] = []
    node_vector_targets: list[str] = []
    graph_classification_adabin_targets: "list[MulticlassClassificationTargetConfig]" = []

    graph_scalar_reduction: dict[str, Literal["sum", "mean"]] = {
    'qst_co2': 'mean',  
    'qst_h2o': 'mean',  
    'qst_n2': 'mean',
    'kh_co2': 'mean', 
    'kh_h2o': 'mean', 
    'kh_n2': 'mean', 
    'selectivity_co2_h2o': 'mean', 
    'selectivity_co2_n2': 'mean', 
    'co2_uptake': 'mean', 
    'n2_uptake': 'mean'
    }


    mof_feature_fusion: dict = {}
    """
    Configuration for fusing global MOF descriptors (PLD, LCD, etc.)
    into the EquiformerV2 model via early or late fusion.
    """
    
    
    log_task_losses: bool = True
    """Log the loss for each task."""
    log_task_steps_and_epochs: bool = True
    """Log the number of steps and epochs for each task."""
    task_loss_scaling: bool = False
    """Scale loss according to the number of task-specific graphs in a batch"""

    tasks: list[TaskConfig]
    """List of datasets/tasks to val/test on."""
    train_tasks: list[TaskConfig]
    """List of datasets/tasks to train on."""
    mt_dataset: MTDatasetConfig = MTDatasetConfig(
        taskify_keys_graph=[],
        taskify_keys_node=[],
        balanced=True,
        strict=True,
    )
    """Configuration for the multi-task dataset."""

    exclude_keys: list[str] = [
        # the following are present in mof-db-1 only
        "qst_n2",  
        "kh_n2",  
        "selectivity_co2_n2",
        "n2_uptake",
        "CSD_ID",
        "Topology"
    ]
    """Keys to exclude when creating a batch from a data list."""

    use_balanced_batch_sampler: bool = True
    """
    Whether to use balanced batch sampler.

    This balances the batches across all distributed nodes (i.e., GPUs, TPUs, nodes, etc.)
    to ensure that each batch has an equal number of **atoms** across all nodes.
    """
    pin_memory: bool = True
    """Whether to use pin memory for data loading."""

    # Heteroscedastic regression config
    heteroscedastic: bool = False
    """Enable heteroscedastic regression (predict mean + variance)"""

    hetero_nll_weight: float = 1.0
    """Weight for Gaussian NLL loss"""

    hetero_std_weight: float = 0.1
    """Weight for auxiliary std supervision loss"""

    hetero_min_var: float = 1e-6
    """Minimum variance for numerical stability"""

    std_targets: list[str] = []
    """List of std targets corresponding to graph_scalar_targets (auto-populated)"""

    # Binary classification for high-value detection
    enable_classification: bool = False
    """Enable binary classification head for high-value detection alongside regression"""

    classification_threshold: float = 0.5
    """Threshold for binary classification (in original scale, e.g., 0.5 mol/kg for co2_uptake)"""

    classification_targets: list[str] = []
    """List of regression targets to also predict as binary classification (e.g., ['co2_uptake'])"""

    classification_loss_weight: float = 1.0
    """Weight for classification loss relative to regression loss"""

    # Focal loss parameters for handling class imbalance
    focal_gamma: float = 2.0
    """Focal loss gamma parameter - higher values focus more on hard examples"""

    focal_alpha_pos: float = 0.75
    """Weight for positive class (high uptake) - higher means more penalty for missing positives"""

    focal_alpha_neg: float = 0.25
    """Weight for negative class (low uptake)"""

    # Asymmetric regression loss (quantile/pinball loss)
    asymmetric_regression: bool = False
    """Enable asymmetric regression loss that penalizes under-prediction more"""

    asymmetric_tau: float = 0.9
    """Quantile level for asymmetric loss. tau=0.9 heavily penalizes under-prediction.
    tau=0.5 is symmetric (equivalent to MAE). Higher tau = more conservative predictions."""

    @override
    def __post_init__(self):
        super().__post_init__()


TModel = TypeVar("TModel", bound=ModelWrapperBase, infer_variance=True)

@final
class AdsorptionDbModel(FinetuneModelBase[AdsorptionDbConfig, TModel]):
    targets: list[AdsorptionDbTarget] = [
        'qst_co2', 
        'qst_h2o',  
        'qst_n2',
        'kh_co2', 
        'kh_h2o', 
        'kh_n2', 
        'selectivity_co2_h2o', 
        'selectivity_co2_n2', 
        'co2_uptake', 
        'n2_uptake'
    ]
    @override
    def __init__(self, hparams: AdsorptionDbConfig):
        super().__init__(hparams)

        # small logic to handle the db1-specific keys in exclude_keys
        # TODO: this is temporary until we have all the Geometric properties for the other datasets.
        task_names = [t.name for t in self.config.tasks]
        db1_specific_keys = ["lcd", "pld", "gcd", "unitcell_volume", "density", "asa", "av", "nav", "CSD_ID"]
        is_db1_solo = (len(task_names) == 1 and task_names[0] == "adsorption_db1_merged")

        if not is_db1_solo:
            # If we are mixing datasets or using a dataset that lacks these keys,
            # we add them to the exclude list to prevent batching errors.
            for k in db1_specific_keys:
                if k not in self.config.exclude_keys:
                    self.config.exclude_keys.append(k)
       
        self.bin_edges = None
        args = self.config.args
        if args.log_predictions:
            self.epoch_results = {target: {} for target in self.config.graph_scalar_targets}
            
        self._train_dataset_sizes: list[int] | None = None
        if self.config.log_task_steps_and_epochs:
            task_steps: dict[str, SumMetric] = {}
            for task in self.config.train_tasks:
                metric = SumMetric()
                metric.persistent(True)
                task_steps[task.name] = metric
            self.task_steps = TypedModuleDict(task_steps)
            
        self.train_metrics = MTFinetuneMetrics(
            config=self.config.metrics,
            provider=self.train_metrics_provider,
            cls_provider=None,
            tasks=self.config.train_tasks,
            roi_getter=self._get_roi
        )
        self.val_metrics = MTFinetuneMetrics(
            config=self.config.metrics,
            provider=self.metrics_provider,
            cls_provider=None,
            tasks=self.config.tasks,
            roi_getter=self._get_roi
        )
        self.test_metrics = MTFinetuneMetrics(
            config=self.config.metrics,
            provider=self.metrics_provider,
            cls_provider=None,
            tasks=self.config.tasks,
            roi_getter=self._get_roi
        )
    
    def _get_roi(self, key: str):
        ROI = {
            "qst_co2": (-60.0, -30.0),
            "qst_h2o": (-80.0, -40.0),
            "kh_co2": (0.000316, 1.0),
            "kh_h2o": (0.01, 100.0),
            "co2_uptake": (0.01, 100.0),
            "selectivity_co2_h2o": (0.0, float("inf")),
        }

        if key not in ROI:
            raise KeyError(
                f"Region of Interest (ROI) not defined for target '{key}'. "
                f"Please add it to the ROI dictionary inside _get_roi function."
            )
        return ROI[key]

    @override
    def validate_config(self, config: AdsorptionDbConfig):
        super().validate_config(config)

        for key in config.graph_scalar_targets:
            assert key in self.targets, f"{key} is not a valid AdsorptionDb target"

    @classmethod
    @override
    def config_cls(cls):
        return AdsorptionDbConfig

    @override
    def metric_prefix(self) -> str:
        return "adsorption_db"


    @override
    def process_aint_graph(self, aint_graph: Graph):
        return aint_graph

    def _dist_sampler(self, dataset, *, shuffle: bool, drop_last: bool):
        """
        Use our own DistributedSampler so Lightning doesn't replace it.
        """
        # Important: pass drop_last=True for train so every rank
        # sees the same number of batches.
        return DistributedSampler(
            dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader(self):
        if self.config.args.atom_bucket_batch_sampler:
            return self._make_loader(self.train_dataset(), train=True, base_for_sizes=self._train_dataset_base)
        return super().train_dataloader()

    def val_dataloader(self):
        if self.config.args.atom_bucket_batch_sampler:
            return self._make_loader(self.val_dataset(), train=False, base_for_sizes=self._val_dataset_base)
        return super().val_dataloader()
    
    def test_dataloader(self):
        if self.config.args.atom_bucket_batch_sampler:
            return self._make_loader(self.test_dataset(), train=False, base_for_sizes=self._test_dataset_base)
        return super().test_dataloader()


    def _get_run_dir(self, trainer) -> str:
        # Prefer logger's experiment directory (W&B exposes .dir)
        if hasattr(trainer.logger, "experiment"):
            exp = trainer.logger.experiment
            if hasattr(exp, "dir") and isinstance(exp.dir, str) and exp.dir:
                return exp.dir

        # Trainer-run directory if available
        if getattr(trainer, "log_dir", None):
            return trainer.log_dir

        # TensorBoard-style: save_dir/name/version_X
        if all(hasattr(trainer.logger, a) for a in ["save_dir", "name", "version"]):
            name = trainer.logger.name or "lightning_logs"
            version = trainer.logger.version
            return os.path.join(trainer.logger.save_dir, name, f"version_{version}")

        # Generic fallback: default_root_dir/lightning_logs/<version>
        base = trainer.default_root_dir or os.getcwd()
        version = getattr(trainer.logger, "version", "unknown")
        return os.path.join(base, "lightning_logs", str(version))

    def _get_results_dir(self, trainer) -> str:
        # Always write predictions outside wandb run dir
        base = trainer.default_root_dir or os.getcwd()
        run_id = getattr(trainer.logger, "version", "unknown")
        out = os.path.join(base, "predictions", f"version_{run_id}")
        os.makedirs(out, exist_ok=True)
        return out

    def _get_results_file(self, trainer, target: str, stage: str) -> str:
        out_dir = self._get_results_dir(trainer)
        model_name = self.config.args.model_name + ("_autoreg" if "autoreg" in self.config.args.checkpoint_tag else "")
        is_pretrained = "" if self.config.args.scratch else "_pretrained"
        misc = f"_BS_{self.config.args.batch_size}"
        if self.config.args.number_of_samples:
            misc += f"_N_{self.config.args.number_of_samples}"
        fname = f"{model_name}_{self.config.args.dataset_name}_{target}{misc}{is_pretrained}_{stage}.csv.gz"
        return os.path.join(out_dir, fname)

    def _task_save_key(self, task: TaskConfig) -> str:
        name = getattr(task, "name", "task")
        if "adsorption_db_1" in name: return "DB1"
        if "adsorption_db_2" in name: return "DB2"
        if "adsorption_cof_db2" in name: return "COF2"
        return name  # fallback to the task name



    @torch.no_grad()
    def step_prediction_logging(self, batch, preds, stage: str):
        # only rank0, only when logging is enabled, and not during sanity checking
        if not self._should_log_now():
            return

        preds_cpu = {
            t: preds[t].detach().to("cpu", non_blocking=True).contiguous()
            for t in self.config.graph_scalar_targets
            if t in preds
        }

        if self.config.heteroscedastic:
            for t in self.config.graph_scalar_targets:
                log_var_key = f"{t}_log_var"
                if log_var_key in preds:
                    preds_cpu[log_var_key] = preds[log_var_key].detach().to("cpu", non_blocking=True).contiguous()

        if hasattr(batch, "sid"):
            mof_id = batch.sid.detach().cpu().tolist() if torch.is_tensor(batch.sid) else list(batch.sid)
        else:
            some = next(iter(preds_cpu.values()))
            mof_id = list(range(int(some.shape[0])))

        def _as_np(x):
            if isinstance(x, (int, float)): return np.asarray(x, dtype=np.float64)
            if torch.is_tensor(x):           return x.detach().cpu().numpy()
            return np.asarray(x)

        # mask: (B, T) -> per task
        task_mask_all = getattr(batch, "task_mask", None)
        if task_mask_all is None:
            task_mask_all = torch.ones((len(mof_id), len(self.config.tasks)), dtype=torch.bool)

        for task_idx, task in enumerate(self.config.tasks):
            mask = task_mask_all[..., task_idx].detach().cpu().numpy().astype(bool).reshape(-1)
            task_key = self._task_save_key(task)
            mof_id_arr = np.asarray(mof_id, dtype=object)

            for target in self.config.graph_scalar_targets:
                if target not in preds_cpu or not hasattr(batch, target):
                    continue

                # shape (B,) after selecting task
                pred_np = preds_cpu[target][..., task_idx].numpy().reshape(-1)
                gt_np   = getattr(batch, target)[..., task_idx].detach().cpu().numpy().reshape(-1)

                pred_np = pred_np[mask]
                gt_np   = gt_np[mask]
                ids_np  = mof_id_arr[mask]

                # pure-CPU denorm, 
                norm = None
                if task.normalization and target in task.normalization:
                    norm = task.normalization[target]
                if norm is not None:
                    mean = _as_np(getattr(norm, "mean", norm.get("mean") if isinstance(norm, dict) else None))
                    std  = _as_np(getattr(norm, "std",  norm.get("std")  if isinstance(norm, dict) else None))
                    if mean is not None and std is not None:
                        pred_np = pred_np * std + mean
                        gt_np   = gt_np   * std + mean
                    ntype = (getattr(norm, "normalization_type", None)
                            or (norm.get("normalization_type") if isinstance(norm, dict) else None)
                            or getattr(norm, "type", None))
                    if ntype == "log":
                        pred_np = np.exp(pred_np)
                        gt_np   = np.exp(gt_np)

                if self.config.heteroscedastic:
                    log_var_key = f"{target}_log_var"
                    if log_var_key in preds_cpu:
                        log_var_np = preds_cpu[log_var_key][..., task_idx].numpy().reshape(-1)
                        log_var_np = log_var_np[mask]
                        pred_std_np = np.exp(0.5 * log_var_np)
                        rows = [
                            {"mof_id": m, "ground_truth": g, "predicted_value": p, "predicted_std": s}
                            for m, g, p, s in zip(ids_np.tolist(), gt_np.tolist(), pred_np.tolist(), pred_std_np.tolist())
                        ]
                    else:
                        rows = [
                            {"mof_id": m, "ground_truth": g, "predicted_value": p}
                            for m, g, p in zip(ids_np.tolist(), gt_np.tolist(), pred_np.tolist())
                        ]
                else:
                    rows = [
                        {"mof_id": m, "ground_truth": g, "predicted_value": p}
                        for m, g, p in zip(ids_np.tolist(), gt_np.tolist(), pred_np.tolist())
                    ]
                store = (
                    self.epoch_results
                        .setdefault(target, {})
                        .setdefault(stage, {})
                        .setdefault(task_key, [])
                )
                store.extend(rows)


    def save_prediction_logging(self, stage: str, trainer=None):
        if stage not in ("val", "test"):
            return
        if not getattr(self.config.args, "log_predictions", False):
            return

        trainer = trainer or self.trainer

        if getattr(trainer, "global_rank", 0) != 0:
            for target in self.config.graph_scalar_targets:
                stage_store = self.epoch_results.get(target, {}).get(stage, {})
                if isinstance(stage_store, dict):
                    for k in list(stage_store.keys()):
                        stage_store[k] = []
            return

        out_root = self._get_results_dir(trainer)
        final_root = os.path.join(out_root, "final", stage)

        for target in self.config.graph_scalar_targets:
            per_task = self.epoch_results.get(target, {}).get(stage, {})
            if not per_task:
                continue

            for task_key, rows in per_task.items():
                if not rows:
                    continue

                sid = np.asarray([r["mof_id"] for r in rows], dtype=object)
                gt  = np.asarray([r["ground_truth"] for r in rows], dtype=np.float32)
                pr  = np.asarray([r["predicted_value"] for r in rows], dtype=np.float32)

                out_dir = os.path.join(final_root, task_key)
                os.makedirs(out_dir, exist_ok=True)
                np.savez_compressed(os.path.join(out_dir, f"{target}.npz"),
                                    sid=sid, gt=gt, pred=pr)
                per_task[task_key] = []

    @override
    def on_train_epoch_start(self):
        self.init_prediction_logging("train")

    @override
    def on_train_epoch_end(self):
        # self._barrier_all() # we dont save train
        pass

    @override
    def on_validation_epoch_start(self):
        self.init_prediction_logging("val")

    @override
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self.save_prediction_logging("val", trainer=self.trainer)
        # self._barrier_all()

    @override
    def on_test_epoch_start(self):
        self.init_prediction_logging("test")

    @override
    def test_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(prefix=f"test/{self.metric_prefix()}/"):
            preds = self(batch)
            self.step_prediction_logging(batch, preds, "test")  # rank>0: no-op
            # Use test metrics here (not val):
            self.log_dict(self.test_metrics(batch, preds))
            torch.cuda.empty_cache()

    
    @override
    def on_test_epoch_end(self):
        self.save_prediction_logging("test")
        # self._barrier_all()

    def on_save_checkpoint(self, checkpoint: dict):
        if self.bin_edges:
            checkpoint["bin_edges"] = [t.cpu() for t in self.bin_edges]

    def on_load_checkpoint(self, checkpoint: dict):
        _bin_edges = checkpoint.get("bin_edges")
        if _bin_edges is not None:
            self.bin_edges = [t.to(self.device) for t in _bin_edges]

    
    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)
        
        # TODO maybe add a warning log?
        # removing the deepcopy is unsafe, but yield around 2x training speed improvement. 
        # the data_transform(data) currenlty returns the data itself, without any modification.
        # we mutatate data.tags and data.cell, which will not effect metric calculations later.

        # data = copy.deepcopy(data)
        # data.tags = 2 * torch.ones(data.natoms)
        # data.tags = data.tags.long()
        data.tags = torch.full((data.natoms,), 2, dtype=torch.long, device=data.pos.device)
        data.cell = data.cell.unsqueeze(dim=0)
        
        cutoff = self.config.args.cutoff
        # Custom settings for neighbours
        # if data.natoms > 560: # double the mean of the number of dataset atoms.
        #     max_neighbors = 6 # theoretical maximum for 1-hop neighbourhood (cutoff~3.0)
        # else:
        #     max_neighbors = 26 # the theoretical max for 3 hops. If we use cutoff~9.0
        #                     #    than this is the max. It is 27-1.
        if hasattr(self.config.args, "max_neighbors"):
            max_neighbors = self.config.args.max_neighbors
        # max_neighbors = 32
        pbc = True
        if self.config.args.no_pbc:
            pbc = False
        
        match self.config.model_cls:
            case models.EScAIPModelWrapper:
                return data
            
            case models.GemNetModelWrapper:
                data = self.generate_graphs(
                    data,
                    cutoffs=Cutoffs.from_constant(cutoff),
                    max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors), 
                    pbc=pbc,
                )
            case models.EquiformerV2ModelWrapper | models.AutoregLlamaEquiformerV2ModelWrapper:
                data = self.generate_graphs(
                    data,
                    cutoffs=Cutoffs.from_constant(cutoff),
                    max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors), 
                    pbc=pbc,
                )
            case _:
                assert_never(self.config.model_cls)
        


        return data

    def _find_filtered_indices(self, dataset, split) -> list[int] | None:
        """This function is meant to cache the indices of the filtered dataset
        from the csv file directly. It is not implemented yet since the dataset
        is not large enough to warrant its use.
        """
        # try:
        #     if (csv_path := self.config.meta.get("csv_path", None)) is not None:
        #         df = pd.read_csv(csv_path)
        #     if (split_keys_file := self.config.meta.get(f"{split}_keys_file"), None) is not None:
        #         train_keys = np.load(split_keys_file, allow_pickle=True)
        #     assert NotImplementedError(
        #         "Buisness logic for cached indices not implemented yet"
        #     )
            
        # except Exception as e:
        #     print(f"Error in _find_filtered_indices: {e}")
        #     return 
        return None

    # def _filter_predicate(self, is_train: bool = False):
    #     return AdsorptionFilter(
    #         is_train=is_train,
    #         max_natoms=(self.config.args.max_natoms if is_train else None),
    #     )
    def _filter_predicate(self, split: Literal["train", "val", "test"]):
        return AdsorptionFilter(
            is_train=(split == "train"),
            max_natoms=self.config.args.max_natoms,  # applies to all splits
        )


    @override
    def on_train_batch_start(self, batch: BaseData, batch_idx: int):
        if not self.config.log_task_steps_and_epochs:
            return

        assert self._train_dataset_sizes
        task_mask = batch.task_mask  # (b, t)
        task_idx = reduce(task_mask, "b t -> t", "sum")  # (t,)
        for idx, task in enumerate(self.config.train_tasks):
            metric = self.task_steps[task.name]
            metric(task_idx[idx])

            step = metric.compute()
            self.log(f"train/{task.name}/step", step)

            epoch = step / self._train_dataset_sizes[idx]
            self.log(f"train/{task.name}/epoch", epoch)
            
    def _task_idx_onehot(self, task_idx: int, tasks: list[TaskConfig]):
        return F.one_hot(
            torch.tensor([task_idx], device=self.device, dtype=torch.long),
            num_classes=len(tasks),
        ).bool()

    def _task_dataset(
        self, task: TaskConfig, split: Literal["train", "val", "test"]
    ) -> DatasetType | None:
        training = split == "train"
        match split:
            case "train":
                if (config := task.train_dataset) is None:
                    return None
            case "val":
                if (config := task.val_dataset) is None:
                    return None
            case "test":
                if (config := task.test_dataset) is None:
                    return None
            case _:
                assert_never(split)

        dataset = config.create_dataset(split)
        self.validate_dataset(dataset)      
        
        # First: filter (before sampling to ensure sampling from valid data only)
        # if (predicate := self._filter_predicate(is_train=training)) is not None:
        if (predicate := self._filter_predicate(split)) is not None:
            dataset = DT.filter_transform(
                dataset, 
                predicate=predicate, 
                filtered_indices=self._find_filtered_indices(dataset, split)
            )

        # Second: sample N from the filtered dataset
        dataset = wrap_common_dataset(dataset, config)

        _transform_map: dict[str, Callable[..., BaseData]] = {
            "adsorption_db_1": _identity_transform,
            "adsorption_db_2": _identity_transform,
            "adsorption_cof_db2": _identity_transform,
        }
        transform_fn = _transform_map.get(task.name, _identity_transform)
        transform = cast(Callable[[BaseData], BaseData],
                         partial(transform_fn, training=training))

        # Last: normalize
        if training:
            DT.set_task_otf_normalization_config(
                dataset, self, self.config.args.targets, task
            )
        if task.normalization:
            transform = T.compose([transform, T.normalize(task.normalization)])
        else:
            pass # TODO
        dataset = DT.transform(dataset, transform)

        return dataset

    def _construct_fm_datasets(self, split: Literal["train", "val", "test"]):
        datasets = []
        for task in self.config.tasks:
            if (dataset := self._task_dataset(task, split=split)) is not None:
                datasets.append(dataset)
        return datasets
    
    @cache
    def train_dataset(self):
        datasets = self._construct_fm_datasets(split="train")
        self._train_dataset_sizes = [len(d) for d in datasets]

        # build the multi-task dataset (untransformed)
        mt = MTSampledDataset(
            datasets,
            self.config.mt_dataset,
            ignore_balancing=False,
            num_tasks=len(self.config.tasks),
        )
        self._train_dataset_base = mt

        dataset = DT.transform(mt, self.train_data_transform)
        return dataset

    def representative_batch_for_testing(self, *, n: int, start_index: int = 0):
        dataset = self.train_dataset()
        data_list = dataset.representative_batch_for_testing(
            n=n, start_index=start_index
        )
        data_list = [self.train_data_transform(data) for data in data_list]
        return data_list

    @cache
    def val_dataset(self):
        datasets = self._construct_fm_datasets(split="val")
        mt = MTSampledDataset(
            datasets,
            self.config.mt_dataset,
            ignore_balancing=True,
            num_tasks=len(self.config.tasks),
        )
        self._val_dataset_base = mt
        return DT.transform(mt, self.val_data_transform)
    
    @cache
    def test_dataset(self):
        datasets = self._construct_fm_datasets(split="test")
        mt = MTSampledDataset(
            datasets,
            self.config.mt_dataset,
            ignore_balancing=True,
            num_tasks=len(self.config.tasks),
        )
        self._test_dataset_base = mt
        return DT.transform(mt, self.test_data_transform)

    def collate_fn(self, data_list: list[BaseData]):
        return Batch.from_data_list(data_list, exclude_keys=self.config.exclude_keys)

    def _task_config(self, name: str):
        return next((task for task in self.config.tasks if task.name == name), None)

    def train_data_transform(self, data: BaseData):
        data = self.data_transform(data)
        return data

    def val_data_transform(self, data: BaseData):
        data = self.data_transform(data)
        return data
    
    def test_data_transform(self, data: BaseData):
        data = self.data_transform(data)
        return data

    def construct_graph_scalar_output_head(self, target: str):
        """Override to support heteroscedastic mode."""
        return self.model.GraphScalarOutputHead(
            self.config,
            self.backbone,
            reduction=self.config.graph_scalar_reduction.get(
                target, self.config.graph_scalar_reduction_default
            ),
            heteroscedastic=self.config.heteroscedastic,
        )

    def construct_binary_classification_output_head(self, target: str):
        """Construct binary classification head for high-value detection."""
        return self.model.GraphBinaryClassificationHead(
            self.config,
            self.backbone,
            reduction=self.config.graph_scalar_reduction.get(
                target, self.config.graph_scalar_reduction_default
            ),
        )

    def construct_output_heads(self):
        self.graph_outputs = TypedModuleDict(
            {
                target: TypedModuleList(
                    [
                        self.construct_graph_scalar_output_head(target)
                        for _ in range(1)
                        # single head for all tasks instead of len(self.config.tasks)
                        # heads with shared weights
                    ]
                )
                for target in self.config.graph_scalar_targets
            },
            key_prefix="ft_mlp_",
        )
        self.graph_classification_outputs = TypedModuleDict(
            {
                target.name: TypedModuleList(
                    [
                        self.construct_graph_classification_output_head(target)
                        for _ in range(1)
                    ]
                )
                for target in (self.config.graph_classification_targets + self.config.graph_classification_adabin_targets)
            },
            key_prefix="ft_mlp_",
        )
        self.node_outputs = TypedModuleDict(
            {
                target: TypedModuleList(
                    [
                        self.construct_node_vector_output_head(target)
                        for _ in range(1)
                    ]
                )
                for target in self.config.node_vector_targets
            },
            key_prefix="ft_mlp_",
        )

        # Binary classification heads for high-value detection
        if self.config.enable_classification and self.config.classification_targets:
            self.binary_classification_outputs = TypedModuleDict(
                {
                    f"{target}_cls": TypedModuleList(
                        [
                            self.construct_binary_classification_output_head(target)
                            for _ in range(1)
                        ]
                    )
                    for target in self.config.classification_targets
                },
                key_prefix="ft_cls_",
            )
        else:
            self.binary_classification_outputs = TypedModuleDict({}, key_prefix="ft_cls_")
        
    @override
    def forward(self, data: BaseData):
        
        match self.config.model_cls:

            case models.GemNetModelWrapper:
                atomic_numbers = data.atomic_numbers - 1
                h = self.embedding(atomic_numbers)  # (N, d_model) # type: ignore[assignment]
                out = cast(self.model.BackboneOutputType, self.backbone(data, h=h)) # type: ignore[assignment]
            
            case (
                models.SchNetModelWrapper | models.EScAIPModelWrapper | 
                models.AutoregLlamaEquiformerV2ModelWrapper |
                models.EquiformerV2ModelWrapper
                ):
                out = cast(self.model.BackboneOutputType, self.backbone(data)) # type: ignore[assignment]

            case _:
                assert_never(self.config.model_cls)
            
        
        output_head_input: OutputHeadInput = {
                    "backbone_output": out,
                    "data": data,
                }

        # Handle heteroscedastic mode: each head returns (mean, log_var) tuple
        if self.config.heteroscedastic:
            preds = {}
            for target, modules in self.graph_outputs.items():
                # Each module returns (mean, log_var) tuple
                # Since we have 1 module shared across tasks, repeat for each task
                results = [module(output_head_input) for module in modules]
                # results is list of (mean, log_var) tuples (typically length 1)
                mean_list = [r[0] for r in results]
                log_var_list = [r[1] for r in results]
                # Repeat for each task and stack
                means = torch.stack(len(self.config.tasks) * mean_list, dim=-1)
                log_vars = torch.stack(len(self.config.tasks) * log_var_list, dim=-1)
                preds[target] = means
                preds[f"{target}_log_var"] = log_vars
            # Add classification outputs (unchanged)
            for target, modules in self.graph_classification_outputs.items():
                preds[target] = torch.stack(
                    len(self.config.tasks) * [module(output_head_input) for module in modules], dim=-1
                )
            # Add node outputs (unchanged)
            for target, modules in self.node_outputs.items():
                preds[target] = torch.stack(
                    len(self.config.tasks) * [module(output_head_input) for module in modules], dim=-1
                )
            # Add binary classification outputs
            if self.config.enable_classification:
                for target, modules in self.binary_classification_outputs.items():
                    preds[target] = torch.stack(
                        len(self.config.tasks) * [module(output_head_input) for module in modules], dim=-1
                    )
        else:
            # Original behavior
            preds = {
                **{
                    target: torch.stack(
                        len(self.config.tasks)*[module(output_head_input) for module in modules], dim=-1
                    )
                    for target, modules in self.graph_outputs.items()
                },
                **{
                    target: torch.stack(
                         len(self.config.tasks)*[module(output_head_input) for module in modules], dim=-1
                    )
                    for target, modules in self.graph_classification_outputs.items()
                },
                **{
                    target: torch.stack(
                         len(self.config.tasks)*[module(output_head_input) for module in modules], dim=-1
                    )
                    for target, modules in self.node_outputs.items()
                },
            }
            # Add binary classification outputs
            if self.config.enable_classification:
                for target, modules in self.binary_classification_outputs.items():
                    preds[target] = torch.stack(
                        len(self.config.tasks) * [module(output_head_input) for module in modules], dim=-1
                    )

        return preds
    
    @staticmethod
    def _safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        b = b.masked_fill(b == 0.0, 1.0)
        return a / b

    def _reduce_loss(
        self,
        loss: torch.Tensor,
        mask: torch.Tensor,
        reduction: Literal["sum", "mean"],
    ):
        match reduction:
            case "sum":
                loss = reduce(loss, "b t -> ", "sum")
            case "mean":
                # loss = reduce(loss, "b t -> ", "sum") / reduce(mask, "b t -> ", "sum")
                loss = self._safe_divide(
                    reduce(loss, "b t -> ", "sum"),
                    reduce(mask, "b t -> ", "sum"),
                )
            case _:
                raise ValueError(f"Unknown redution: {reduction}")

        return loss

    def _gaussian_nll_loss(
        self,
        y_pred_mean: torch.Tensor,
        y_pred_log_var: torch.Tensor,
        y_true: torch.Tensor,
        min_var: float = 1e-6,
    ) -> torch.Tensor:
        """
        Gaussian Negative Log-Likelihood loss.
        Args:
            y_pred_mean: Predicted mean (B, T)
            y_pred_log_var: Predicted log-variance (B, T)
            y_true: Ground truth (B, T)
            min_var: Minimum variance for stability

        Returns:
            NLL loss per sample (B, T)
        """
        var = torch.exp(y_pred_log_var).clamp(min=min_var)

        nll = 0.5 * (y_pred_log_var + (y_true - y_pred_mean).pow(2) / var)

        return nll

    def _std_supervision_loss(
        self,
        y_pred_log_var: torch.Tensor,
        y_true_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auxiliary supervision loss on predicted std vs known std.

        Args:
            y_pred_log_var: Predicted log-variance (B, T)
            y_true_std: Ground truth std (B,) or (B, T), already log-normalized

        Returns:
            L1 loss in log-space per sample (B, T)
        """
        pred_log_std = 0.5 * y_pred_log_var

        true_log_std = y_true_std

        if true_log_std.dim() == 1 and pred_log_std.dim() == 2:
            true_log_std = true_log_std.unsqueeze(-1).expand_as(pred_log_std)

        return F.l1_loss(pred_log_std, true_log_std, reduction="none")

    def _compute_calibration_metrics(
        self,
        batch: BaseData,
        preds: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute calibration metrics for heteroscedastic predictions.

        A well-calibrated uncertainty estimator should have:
        1. Predicted std correlating with actual absolute error
        2. Errors falling within predicted confidence intervals

        Returns dict of metrics to log.
        """
        if not self.config.heteroscedastic:
            return {}

        metrics = {}
        for target in self.config.graph_scalar_targets:
            if target not in preds or f"{target}_log_var" not in preds:
                continue

            y_pred = preds[target]
            y_pred_log_var = preds[f"{target}_log_var"]
            y_true = getattr(batch, target, None)

            if y_true is None:
                continue

            # Flatten across tasks (take mean across task dim for simplicity)
            task_mask = batch.task_mask
            # Average across tasks where mask is valid
            y_pred_flat = (y_pred * task_mask).sum(dim=-1) / task_mask.sum(dim=-1).clamp(min=1)
            y_true_flat = (y_true * task_mask).sum(dim=-1) / task_mask.sum(dim=-1).clamp(min=1)
            log_var_flat = (y_pred_log_var * task_mask).sum(dim=-1) / task_mask.sum(dim=-1).clamp(min=1)

            # Compute predicted std and absolute error
            pred_std = torch.exp(0.5 * log_var_flat)
            abs_error = (y_pred_flat - y_true_flat).abs()

            # Spearman correlation (approximated via Pearson on ranks)
            # Higher correlation means better calibration
            if len(pred_std) > 2:
                # Rank transform
                pred_std_rank = pred_std.argsort().argsort().float()
                abs_error_rank = abs_error.argsort().argsort().float()

                # Pearson correlation of ranks = Spearman
                pred_std_centered = pred_std_rank - pred_std_rank.mean()
                abs_error_centered = abs_error_rank - abs_error_rank.mean()

                numer = (pred_std_centered * abs_error_centered).sum()
                denom = (pred_std_centered.pow(2).sum() * abs_error_centered.pow(2).sum()).sqrt()
                spearman = numer / denom.clamp(min=1e-8)
                metrics[f"{target}_calib_spearman"] = spearman

            # ENCE: Expected Normalized Calibration Error
            # Check if errors fall within 1-sigma (68%) and 2-sigma (95%) intervals
            within_1sigma = (abs_error <= pred_std).float().mean()
            within_2sigma = (abs_error <= 2 * pred_std).float().mean()
            metrics[f"{target}_within_1sigma"] = within_1sigma  # Should be ~0.68
            metrics[f"{target}_within_2sigma"] = within_2sigma  # Should be ~0.95

            # Mean absolute calibration error
            # Compare predicted std with actual error magnitude
            mace = (pred_std - abs_error).abs().mean()
            metrics[f"{target}_mace"] = mace

        return metrics

    def _graph_level_loss(
        self,
        batch: BaseData,
        preds: dict[str, torch.Tensor],
        target: str,
        scale: list[float],
        tasks: list[TaskConfig],
        reduction: Literal["sum","mean"] = "mean",
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        mask = batch.task_mask

        args = self.config.args

        # HETEROSCEDASTIC BRANCH
        if self.config.heteroscedastic:
            y_pred_mean = preds[target]
            y_pred_log_var = preds[f"{target}_log_var"]
            y_true = batch[target].to(y_pred_mean.dtype)

            nll_loss = self._gaussian_nll_loss(
                y_pred_mean, y_pred_log_var, y_true,
                min_var=self.config.hetero_min_var,
            )

            std_target_name = f"{target}_std"
            has_std_supervision = hasattr(batch, std_target_name)
            if has_std_supervision:
                y_true_std = getattr(batch, std_target_name).to(y_pred_log_var.dtype)

                std_norm_mean = getattr(batch, f"{std_target_name}_norm_mean", None)
                std_norm_std = getattr(batch, f"{std_target_name}_norm_std", None)
                if std_norm_mean is not None and std_norm_std is not None:
                    y_true_std = y_true_std * std_norm_std + std_norm_mean

                std_loss = self._std_supervision_loss(
                    y_pred_log_var, y_true_std
                )
                target_loss = (
                    self.config.hetero_nll_weight * nll_loss +
                    self.config.hetero_std_weight * std_loss
                )
            else:
                std_loss = None
                target_loss = self.config.hetero_nll_weight * nll_loss

            # Log heteroscedastic-specific metrics during training
            if self.training:
                # Log individual loss components (reduced)
                nll_reduced = self._reduce_loss(nll_loss * mask, mask, reduction)
                self.log(f"{target}_nll_loss", nll_reduced)
                if has_std_supervision and std_loss is not None:
                    std_reduced = self._reduce_loss(std_loss * mask, mask, reduction)
                    self.log(f"{target}_std_loss", std_reduced)

                # Log predicted std statistics
                pred_std = torch.exp(0.5 * y_pred_log_var)  # Convert log_var to std
                self.log(f"{target}_pred_std_mean", pred_std.mean())
                self.log(f"{target}_pred_std_min", pred_std.min())
                self.log(f"{target}_pred_std_max", pred_std.max())

            y_pred = y_pred_mean

        # ORIGINAL BRANCH
        else:
            y_pred = preds[target]
            y_true = batch[target].to(y_pred.dtype)

            # Asymmetric regression loss (quantile/pinball) for discovery tasks
            if self.config.asymmetric_regression:
                target_loss = self._asymmetric_regression_loss(
                    y_pred, y_true, tau=self.config.asymmetric_tau
                )
            elif args.loss == "l2":
                target_loss = F.mse_loss(y_pred, y_true, reduction="none")
            elif args.loss == "l1":
                target_loss = F.l1_loss(y_pred, y_true, reduction="none")
            elif args.loss == "huber":
                target_loss = F.smooth_l1_loss(y_pred, y_true, reduction="none", beta=1.0)
            else:
                raise ValueError(f"Unknown loss: {args.loss}")

        # invfreq weighting loss
        # the default values are set to non-skewed distribution
        if getattr(args, "invfreq", False):
            ramp = self._ramp(True, getattr(args, "invfreq_warmup_epochs", 0))
            if ramp > 0.0:
                w_inv = self._invfreq_weights(
                    y_true, 
                    float(getattr(args, "invfreq-beta", 0.75)),
                    int(getattr(args, "invfreq_bins", 50)),
                )
                target_loss = target_loss * (1.0 - ramp + ramp * w_inv)

        
        # focal regression loss
        # the default values are set to non-skewed distribution
        if getattr(args, "focal", False):
            ramp = self._ramp(True, getattr(args, "focal_warmup_epochs", 0))
            if ramp > 0.0:
                w_focal = self._focal_weights(
                    y_pred, y_true, mask,
                    gamma=float(getattr(args, "focal_gamma", 1.0)),
                    eps=float(getattr(args, "focal_eps", 1e-6)),
                    grad_through=bool(getattr(args, "focal_grad_through", False)),
                    normalize=True,
                )
                # smooth turn-on: loss := ( (1-ramp)*1 + ramp*w ) * loss
                target_loss = target_loss * (1.0 - ramp + ramp * w_focal)

        # ---------------- ROI penalty ----------------
        roi_penalty = getattr(self.config.args, "roi_penalty", 1.0)
        if roi_penalty != 1.0:
            try:
                low, high = self._get_roi(target)

                # Use ground truth position in ROI
                norm = None
                for task in tasks:
                    if task.normalization and target in task.normalization:
                        norm = task.normalization[target]
                        break

                if norm is not None:
                    gt_denorm = T.denormalize_tensor(y_true, norm)
                else:
                    gt_denorm = y_true

                roi_mask = (gt_denorm >= low) & (gt_denorm <= high)

                # Penalize only ROI region errors
                target_loss = target_loss * torch.where(
                    roi_mask,
                    torch.tensor(roi_penalty, device=target_loss.device),
                    torch.tensor(1.0, device=target_loss.device),
                )
            except KeyError:
                pass
        # ------------------------------------------------


        scale_tensor = torch.tensor(scale, device=self.device, dtype=target_loss.dtype)
        target_loss = target_loss * scale_tensor.unsqueeze(0)

        mask = batch.task_mask
        target_loss = target_loss.masked_fill(~mask, 0.0)

        target_loss = target_loss.masked_fill(~mask, 0.0)

        need_task_scalars = self.config.log_task_losses or self.config.task_loss_scaling
        task_scalars: list[torch.Tensor] = []

        if need_task_scalars:
            ctx = (nullcontext() if self.config.task_loss_scaling else torch.no_grad())
            with ctx:
                for task_idx, task in enumerate(tasks):
                    task_mask = mask & self._task_idx_onehot(task_idx, tasks)
                    task_loss_mat = target_loss.masked_fill(~task_mask, 0.0)
                    task_scalar = self._reduce_loss(task_loss_mat, task_mask, reduction=reduction)
                    if self.training and self.config.log_task_losses:
                        self.log(
                            f"{task.name}/{target}_loss_scaled",
                            task_scalar,
                        )
                    task_scalars.append(task_scalar)

        return target_loss, mask, task_scalars


    @override
    def compute_losses(
        self,
        batch: BaseData,
        preds: dict[str, torch.Tensor],
        tasks: list[TaskConfig],
    ) -> torch.Tensor:
        per_target_losses: list[torch.Tensor] = []
        # compatible with the new _graph_level_loss
        # if task_loss_scaling is used, it can handle 
        # the separate losses
        for target in self.config.graph_scalar_targets:
            coef = [
                task.graph_scalar_loss_coefficients.get(
                    target, self.config.graph_scalar_loss_coefficient_default
                )
                for task in tasks
            ]
            reduction_kind = self.config.graph_scalar_reduction.get(target, "mean")

            target_loss, target_loss_mask, task_scalars = self._graph_level_loss(
                batch, preds, target, coef, tasks, reduction=reduction_kind
            )

            if self.config.task_loss_scaling:
                # python sum() starts from  int 0. 
                # toch.stack sum is save here.
                loss_t = (torch.stack(task_scalars).sum()
                        if task_scalars else torch.tensor(0.0, device=self.device))
            else:
                loss_t = self._reduce_loss(target_loss, target_loss_mask, reduction=reduction_kind)

            if self.training:
                self.log(
                    f"{target}_loss_scaled", loss_t,
                )
            per_target_losses.append(loss_t)

        # Binary classification loss for high-value detection
        if self.config.enable_classification and self.config.classification_targets:
            for target in self.config.classification_targets:
                cls_key = f"{target}_cls"
                if cls_key not in preds:
                    continue

                cls_logits = preds[cls_key]  # (B, T)
                regression_targets = batch[target].to(cls_logits.dtype)  # (B, T)

                # Get normalization for this target to compute binary labels
                norm = None
                for task in tasks:
                    if task.normalization and target in task.normalization:
                        norm = task.normalization[target]
                        break

                # Compute binary labels based on threshold
                binary_labels = self._compute_binary_labels(
                    regression_targets,
                    self.config.classification_threshold,
                    normalization=norm,
                )

                # Compute asymmetric focal loss
                cls_loss = self._asymmetric_focal_bce_loss(
                    cls_logits,
                    binary_labels,
                    gamma=self.config.focal_gamma,
                    alpha_pos=self.config.focal_alpha_pos,
                    alpha_neg=self.config.focal_alpha_neg,
                )

                # Apply task mask
                mask = batch.task_mask
                cls_loss = cls_loss.masked_fill(~mask, 0.0)
                cls_loss_reduced = self._reduce_loss(cls_loss, mask, reduction="mean")

                # Apply classification loss weight
                cls_loss_weighted = self.config.classification_loss_weight * cls_loss_reduced

                if self.training:
                    self.log(f"{target}_cls_loss", cls_loss_reduced)

                    # Log classification metrics
                    with torch.no_grad():
                        cls_probs = torch.sigmoid(cls_logits)
                        cls_preds = (cls_probs > 0.5).float()

                        # Flatten across tasks using mask
                        flat_preds = cls_preds[mask]
                        flat_labels = binary_labels[mask]

                        if flat_labels.sum() > 0:
                            # True positive rate (recall for positive class)
                            tp = ((flat_preds == 1) & (flat_labels == 1)).float().sum()
                            fn = ((flat_preds == 0) & (flat_labels == 1)).float().sum()
                            recall = tp / (tp + fn + 1e-8)
                            self.log(f"{target}_cls_recall", recall)

                            # Precision
                            fp = ((flat_preds == 1) & (flat_labels == 0)).float().sum()
                            precision = tp / (tp + fp + 1e-8)
                            self.log(f"{target}_cls_precision", precision)

                        # Class distribution
                        pos_frac = flat_labels.mean()
                        self.log(f"{target}_cls_pos_fraction", pos_frac)

                per_target_losses.append(cls_loss_weighted)

        total_loss = (torch.stack(per_target_losses).sum()
                    if per_target_losses else torch.tensor(0.0, device=self.device))
        if self.training:
            self.log(
                "loss", total_loss
            )
        return total_loss

    @override
    def training_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(prefix=f"train/{self.metric_prefix()}/"):
            preds = self(batch)

            loss = self.compute_losses(batch, preds, self.config.train_tasks)
            self.log_dict(self.train_metrics(batch, preds))


            torch.cuda.empty_cache()

            return loss
        
    @override
    def validation_step(self, batch, batch_idx):
        with self.log_context(prefix=f"val/{self.metric_prefix()}/"):

            preds = self(batch)
            self.step_prediction_logging(batch, preds, "val")
            val_loss = self.compute_losses(batch, preds, self.config.tasks)
            batch_size = int(batch.batch.max()) + 1
            self.log(f"loss", val_loss, batch_size=batch_size, prog_bar=True)
            self.log_dict(self.val_metrics(batch, preds))

            # Log calibration metrics for heteroscedastic mode
            if self.config.heteroscedastic:
                calib_metrics = self._compute_calibration_metrics(batch, preds)
                if calib_metrics:
                    self.log_dict(calib_metrics, batch_size=batch_size)

            
    def train_metrics_provider(
        self,
        prop: str,
        batch: BaseData,
        preds: dict[str, torch.Tensor],
        task_idx: int,
    ) -> MetricPair | None:
        # Skip variance predictions from heteroscedastic mode
        if "_log_var" in prop:
            return None

        if (pred := preds.get(prop)) is None or (
            target := getattr(batch, prop, None)
        ) is None:
            return None

        task_mask = batch.task_mask[..., task_idx]  # (b,)
        pred = pred[..., task_idx][task_mask]      # (b, t) -> (b,)
        target  = target[...,  task_idx][task_mask]
        task = self.config.train_tasks[task_idx]
        if task.normalization and prop in task.normalization:
            norm = task.normalization[prop]
            pred = T.denormalize_tensor(pred, norm)
            target = T.denormalize_tensor(target, norm)

        return MetricPair(predicted=pred, ground_truth=target)
    
    
    @override
    def metrics_provider(
        self,
        prop: str,
        batch: BaseData,
        preds: dict[str, torch.Tensor],
        task_idx: int,
    ) -> MetricPair | None:
        # Skip variance predictions from heteroscedastic mode
        if "_log_var" in prop:
            return None

        if (pred := preds.get(prop)) is None or (
            target := getattr(batch, prop, None)
        ) is None:
            return None

        task_mask = batch.task_mask[..., task_idx]  # (b,)
        pred = pred[..., task_idx][task_mask]      # (b, t) -> (b,)
        target  = target[...,  task_idx][task_mask]
        task = self.config.tasks[task_idx]

        if task.normalization and prop in task.normalization:
            norm = task.normalization[prop]

            # we need to store the log-norm results before denorm
            setattr(batch, f"{prop}_normalized", pred)
            setattr(batch, f"{prop}_normalized_target", target)

            pred = T.denormalize_tensor(pred, norm)
            target = T.denormalize_tensor(target, norm)

        return MetricPair(predicted=pred, ground_truth=target)

    @override
    def _rlp_metric(self, rlp_cfg: RLPConfig):
        monitor = rlp_cfg.monitor
        assert monitor is not None, "RLP monitor must be specified."

        metric_prefix = f"val/{self.metric_prefix()}/"
        assert monitor.startswith(metric_prefix), f"RLP {monitor=} must start with {metric_prefix}"
        short = monitor[len(metric_prefix):]

        metric = None
        m_store = getattr(self.val_metrics, "_metrics", None)

        if m_store is not None:
            try:
                if short in m_store:
                    metric = m_store[short]
            except Exception:
                if isinstance(m_store, dict):
                    metric = m_store.get(short)

        if metric is not None and hasattr(metric, "compute"):
            return metric

        class _ScalarWrapper:
            def __init__(self, fetch, device):
                self._fetch = fetch
                self._device = device
            def compute(self):
                v = self._fetch()
                if v is None:
                    raise RuntimeError(f"RLP monitor '{monitor}' missing in callback_metrics.")
                return v if isinstance(v, torch.Tensor) else torch.tensor(v, device=self._device)

        return _ScalarWrapper(lambda: self.trainer.callback_metrics.get(monitor), self.device)



    def _unwrap_base(self, ds):
        seen = set()
        while hasattr(ds, "dataset") and id(ds) not in seen:
            seen.add(id(ds))
            ds = ds.dataset
        return ds

    def _collect_natoms_list(self, ds) -> list[int]:
        """
        Try several cheap paths to get a natoms-per-item list without triggering heavy transforms.
        """
        for attr in ("natoms", "natoms_per_item", "sizes"):
            if hasattr(ds, attr):
                try:
                    arr = getattr(ds, attr)
                    return [int(x) for x in list(arr)]
                except Exception:
                    pass

        out = []
        for i in range(len(ds)):
            item = ds[i]
            out.append(int(item.natoms))
        return out

    def _natoms_per_item(self, dataset) -> list[int]:
        """
        Public helper used by _make_loader. It unwraps to base before collecting natoms.
        """
        base = self._unwrap_base(dataset)
        return self._collect_natoms_list(base)

    def _compute_dataset_stats(self, sizes: list[int], dataset, sample_size: int = 100):
        """compute avg_num_nodes and avg_degree
        """
        args = self.config.args
        try:
            avg_nodes = float(np.mean(sizes)) if sizes else float(getattr(self.config.backbone, "avg_num_nodes", 1.0))
            avg_degree = float(getattr(self.config.backbone, "avg_degree", 1.0))

            did_degree = False
            if getattr(args, "compute_avg_dataset_stats_degree", True):
                n = min(int(sample_size), len(dataset))
                if n > 0:
                    degs = []
                    cutoff = getattr(args, "cutoff", getattr(self.config.backbone, "max_radius", 5.0))
                    max_neighbors = getattr(self.config.backbone, "max_neighbors", 48)
                    for i in range(n):
                        try:
                            item = dataset[i]
                            g = self.generate_graphs(
                                item,
                                cutoffs=Cutoffs.from_constant(cutoff),
                                max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
                                pbc=not getattr(args, "no_pbc", False),
                            )
                            natoms = int(getattr(g, "natoms", getattr(g, "n_nodes", getattr(g, "pos", None)).shape[0]))
                            if hasattr(g, "edge_index"):
                                src = g.edge_index[0]
                                counts = np.bincount(src.detach().cpu().numpy(), minlength=natoms)
                                degs.append(float(counts.sum() / max(1, natoms)))
                        except Exception:
                            continue
                    if degs:
                        avg_degree = float(np.mean(degs))
                        did_degree = True

            if getattr(self, "computed_dataset_stats", None) is None:
                self.computed_dataset_stats = {}
            self.computed_dataset_stats.update(dict(avg_num_nodes=avg_nodes, avg_degree=avg_degree))

            if getattr(self.config, "backbone", None) is not None:
                try:
                    self.config.backbone.avg_num_nodes = float(avg_nodes)
                    self.config.backbone.avg_degree = float(avg_degree)
                    print(f"[config] set backbone.avg_num_nodes={avg_nodes:.3f}, avg_degree={avg_degree:.3f} "
                        f"(degree_sampled={did_degree})")
                except Exception:
                    print("[config] failed to set backbone dataset stats.")

            if getattr(self.trainer, "global_rank", 0) == 0:
                print(f"[config] dataset stats: avg_num_nodes={avg_nodes:.3f}, "
                    f"avg_degree={avg_degree:.3f}, "
                    f"len={len(dataset)}, sizes_len={len(sizes)}")

        except Exception:
            print("[config] failed to compute dataset stats.")
            return

    def _make_loader(self, dataset, *, train: bool, base_for_sizes=None):
        sizes_src = base_for_sizes if base_for_sizes is not None else dataset
        sizes = self._natoms_per_item(sizes_src)
        assert len(sizes) == len(dataset), "sizes/dataset length mismatch"

        # compute and override dataset statistics used by the b|ackbone.
        # --compute_avg_dataset_stats and --compute_avg_dataset_stats_degree
        # set to true by default
        # if getattr(self.config.args, "compute_avg_dataset_stats", True): 
        sample_size = getattr(self.config.args, "avg_stats_sample_size", 10000)
        # pass the untransformed base dataset for degree sampling if possible
        self._compute_dataset_stats(sizes, sizes_src, sample_size=sample_size)

        args = self.config.args
        cap = getattr(args, "max_atoms_per_batch", None)
        if cap is None:
            cap = int(os.getenv("MAX_ATOMS_PER_BATCH", 0)) or int(np.median(sizes) * max(1, self.config.batch_size))

        sampler_dataset = dataset
        if getattr(self.trainer, "world_size", 1) > 1:
            base_sampler = DistributedSampler(
                sampler_dataset,
                shuffle=train,
                drop_last=True if train else False,
            )
        else:
            base_sampler = RandomSampler(sampler_dataset)

        batch_sampler = AtomBucketBatchSampler(
            base_sampler=base_sampler,
            sizes=sizes,
            max_atoms_per_batch=cap,
            bucket_boundaries=(128, 256, 384, 512, 768, 1024, 1536, 2048),
            drop_last=True if train else False,
            seed=getattr(args, "seed", 0),
        )
        if hasattr(batch_sampler, "__len__") and len(batch_sampler) == 0:
            raise RuntimeError(f"AtomBucketBatchSampler built 0 batches (cap={cap}). "
                            f"Increase cap or set drop_last=False for debugging.")
        num_workers = getattr(args, "num_workers", 2)
        mp_ctx = None
        timeout_s = getattr(args, "dataloader_timeout_s", 0)
        dl_kwargs = dict(
            dataset=dataset,
            batch_sampler=batch_sampler,
            pin_memory=self.config.pin_memory,
            collate_fn=partial(_collate_exclude, exclude_keys=self.config.exclude_keys),
        )
    
        if num_workers > 0:
            dl_kwargs.update(
                num_workers=num_workers,
                persistent_workers=True,  # TODO
                worker_init_fn=_seed_worker,
                multiprocessing_context=mp_ctx,
                timeout=timeout_s,
                prefetch_factor=4,
            )
        else:
            # Single-process loader must have timeout == 0 and no prefetch/worker args.
            dl_kwargs.update(num_workers=0, timeout=0)
    
        return DataLoader(**dl_kwargs)

    def _is_rank0(self):
        return getattr(self.trainer, "global_rank", 0) == 0

    def _should_log_now(self):
        if not self.config.args.log_predictions:
            return False
        if not self._is_rank0():
            return False
        if getattr(self.trainer, "sanity_checking", False):
            return False
        return True

    def init_prediction_logging(self, stage: str):
        if not getattr(self.config.args, "log_predictions", False):
            return
        if not hasattr(self, "epoch_results") or self.epoch_results is None:
            self.epoch_results = {t: {} for t in self.config.graph_scalar_targets}
        for t in self.config.graph_scalar_targets:
            stage_store = self.epoch_results.setdefault(t, {}).setdefault(stage, {})
            for task in self.config.tasks:
                key = self._task_save_key(task)
                stage_store.setdefault(key, [])


    def _barrier_all(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


    def _ramp(self, enabled: bool, warmup_epochs: int) -> float:
        """Linear 0->1 ramp over `warmup_epochs`. Returns 0 if disabled."""
        if not enabled:
            return 0.0
        if warmup_epochs <= 0:
            return 1.0
        return float(min(1.0, (self.current_epoch + 1) / max(1, warmup_epochs)))

    def _invfreq_weights(self, y: torch.Tensor, beta: float, bins: int) -> torch.Tensor:
        """
        y: (B, T) targets in the  normalized training space.
        Returns weights B, T ~ 1 / freq(bin(y))^beta, mean-normalized to 1 per task.
        """
        B, T = y.shape
        device = y.device
        dtype = y.dtype

        edges = torch.linspace(-4, 4, bins + 1, device=device, dtype=dtype)

        w = torch.ones((B, T), device=device, dtype=dtype)
        for t in range(T):
            y_t = y[:, t].detach()
            bin_idx = torch.bucketize(y_t, edges, right=True) - 1
            bin_idx = bin_idx.clamp_(0, bins - 1)

            counts = torch.bincount(bin_idx, minlength=bins).to(dtype) + 1.0
            w[:, t] = counts[bin_idx].reciprocal().pow(beta)

        w = w / (w.mean(dim=0, keepdim=True) + 1e-8)
        return w


    def _focal_weights(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
        eps: float,
        grad_through: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        err = (y_pred - y_true).abs().clamp_min(eps)
        w = err.pow(gamma)
        if not grad_through:
            w = w.detach()
        if normalize:
            den = mask.to(w.dtype).sum(dim=0, keepdim=True).clamp_min(1.0)
            mean_w = (w * mask).sum(dim=0, keepdim=True) / den
            w = w / (mean_w + 1e-12)
        return w

    def _asymmetric_regression_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        tau: float = 0.9,
    ) -> torch.Tensor:
        """
        Asymmetric (Pinball/Quantile) loss for regression.

        Penalizes under-prediction more than over-prediction when tau > 0.5.
        This biases predictions toward the upper quantile, which is what we
        want for discovery (better to flag a false positive than miss a hit).

        Args:
            y_pred: Predictions (B,) or (B, T)
            y_true: Targets (B,) or (B, T)
            tau: Quantile level. tau=0.9 means 90th percentile (heavily penalize under-prediction)
                 tau=0.5 is equivalent to MAE (symmetric)

        Returns:
            Per-sample loss (B,) or (B, T)
        """
        residual = y_true - y_pred
        loss = torch.where(
            residual >= 0,
            tau * residual,           # Under-prediction: weight = tau
            (tau - 1) * residual      # Over-prediction: weight = (1 - tau)
        )
        return loss

    def _asymmetric_focal_bce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha_pos: float = 0.75,
        alpha_neg: float = 0.25,
    ) -> torch.Tensor:
        """
        Asymmetric Focal Binary Cross-Entropy Loss.

        Combines focal loss (down-weight easy examples) with asymmetric weighting
        (penalize false negatives more than false positives).

        Args:
            logits: Raw logits (B,) or (B, T)
            targets: Binary targets (B,) or (B, T), 1 = positive (high uptake)
            gamma: Focal loss focusing parameter
            alpha_pos: Weight for positive class (high uptake)
            alpha_neg: Weight for negative class (low uptake)

        Returns:
            Per-sample loss (B,) or (B, T)
        """
        probs = torch.sigmoid(logits)

        # Binary cross-entropy components
        bce_pos = -targets * torch.log(probs.clamp(min=1e-8))
        bce_neg = -(1 - targets) * torch.log((1 - probs).clamp(min=1e-8))

        # Focal weighting: down-weight confident predictions
        focal_weight_pos = (1 - probs).pow(gamma)  # Hard positives: low prob but should be 1
        focal_weight_neg = probs.pow(gamma)         # Hard negatives: high prob but should be 0

        # Asymmetric weighting: penalize FN (missing high uptake) more
        loss = (
            alpha_pos * focal_weight_pos * bce_pos +
            alpha_neg * focal_weight_neg * bce_neg
        )

        return loss

    def _compute_binary_labels(
        self,
        regression_targets: torch.Tensor,
        threshold: float,
        normalization: Optional[NormalizationConfig] = None,
    ) -> torch.Tensor:
        """
        Convert regression targets to binary labels based on threshold.

        Args:
            regression_targets: Normalized regression targets (B,) or (B, T)
            threshold: Threshold in original scale (e.g., 0.5 mol/kg)
            normalization: Normalization config to denormalize targets

        Returns:
            Binary labels: 1 if above threshold, 0 otherwise
        """
        # Denormalize to original scale for threshold comparison
        if normalization is not None:
            denorm_targets = T.denormalize_tensor(regression_targets, normalization)
        else:
            denorm_targets = regression_targets

        # Create binary labels
        binary_labels = (denorm_targets >= threshold).float()

        return binary_labels
