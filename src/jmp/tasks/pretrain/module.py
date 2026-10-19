"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import fnmatch
from abc import abstractmethod
from collections.abc import Callable
from functools import cache, partial
from logging import getLogger
from typing import Annotated, Generic, Literal, TypeAlias, assert_never, cast, Any
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from einops import pack, rearrange, reduce
from lightning.pytorch.callbacks import ModelCheckpoint
from jmp.lightning import Base, BaseConfig, Field, LightningModuleBase, TypedConfig
from jmp.lightning.data.balanced_batch_sampler import BalancedBatchSampler
from jmp.lightning.util.typed import TypedModuleDict, TypedModuleList
from jmp.tasks.finetune.base import (
    PrimaryMetricConfig,
    CheckpointBestConfig
)
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRSchedulerConfig,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.utils import dropout_edge
from torch_scatter import scatter
from torchmetrics import SumMetric
from typing_extensions import TypeVar, override

from ...datasets.pretrain_lmdb import PretrainDatasetConfig as PretrainDatasetConfigBase
from ...datasets.pretrain_lmdb import PretrainLmdbDataset, PermutedDataset
from ...models.gemnet.backbone import GemNetOCBackbone, GOCBackboneOutput
from ...models.gemnet.config import GOCBackboneConfig
from ...models.equiformer_v2.config import (
    EquiformerV2Config, EquiformerV2BackboneOutput, AutoregressiveLlamaEquiformerV2Config
)
from ...models.gemnet.layers.base_layers import ScaledSiLU
from ...modules import transforms as T
from ...modules.dataset import dataset_transform as DT
from ...modules.dataset.common import CommonDatasetConfig, wrap_common_dataset
from ...modules.dataset.concat_dataset import MTDatasetConfig, MTSampledDataset
from ...modules.ema import EMAConfig
from ...modules.metrics import FMMetrics
from ...modules.scheduler.linear_warmup_cosine_annealing import (
    LinearWarmupCosineAnnealingLR,
    PerParamGroupLinearWarmupCosineAnnealingLR,
)
from ...modules.transforms.normalize import NormalizationConfig, PositionNormalizationConfig
from ...utils.goc_graph import (
    Cutoffs,
    Graph,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)
from ..config import (
    EmbeddingConfig,
    OptimizerConfig,
    OutputConfig,
    optimizer_from_config,
)

from jmp.tasks.pretrain import model_wrapper_base as models
from jmp.tasks.pretrain.model_wrapper_base import (
    ModelWrapperBase,
    Backbone, BackboneConfig
)

log = getLogger(__name__)


class EpochAwareDistributedSampler(DistributedSampler):
    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        dataset = self.dataset
        if isinstance(dataset, MTSampledDataset):
            # If the dataset is a MTSampledDataset, we need to set the epoch on each task dataset
            for task_dataset in dataset.datasets:
                if hasattr(task_dataset, "set_epoch"):
                    task_dataset.set_epoch(epoch)


class LinearWarmupCosineAnnealingSchedulerConfig(TypedConfig):
    name: Literal["linear_warmup_cosine_annealing"] = "linear_warmup_cosine_annealing"

    warmup_steps: int = 0
    max_steps: int | None = None
    max_epochs: int | None = None
    warmup_start_lr_factor: float = 0.0
    min_lr_factor: float = 1.0e-2
    last_step: int = -1

class WarmupCosineSchedulerConfig(TypedConfig):
    name: Literal["linear_warmup_cosine"] = "linear_warmup_cosine"

    warmup_steps: int = 0
    warmup_epochs: int | None = None
    max_steps: int | None = None
    max_epochs: int | None = None
    warmup_start_lr_factor: float = 0.0
    min_lr_factor: float = 1.0e-2
    hold_epochs: int = 0
    last_step: int = -1
    should_restart: bool = False
    
class WarmupCosineLambdaLRSchedulerConfig(TypedConfig):
    name: Literal["linear_warmup_cosine_lambda_lr"] = "linear_warmup_cosine_lambda_lr"

    warmup_epochs: float | None = None
    warmup_factor: float = 0.0
    max_epochs: int | None = None
    lr_min_factor: float = 1.0e-2
    hold_epochs: int = 0
    warmup_steps: int | None = None
    max_steps: int | None = None
    last_step: int = -1

LRSchedulerConfig: TypeAlias = Annotated[
    LinearWarmupCosineAnnealingSchedulerConfig | WarmupCosineSchedulerConfig
    | WarmupCosineLambdaLRSchedulerConfig,
    Field(discriminator="name")
]

class ParamSpecificOptimizerConfig(TypedConfig):
    name: str | None = None
    """The name of the parameter group for this config"""

    paremeter_patterns: list[str] = []
    """List of parameter patterns to match for this config"""

    optimizer: OptimizerConfig | None = None
    """
    The optimizer config for this parameter group.
    If None, the default optimizer will be used.
    """

    lr_scheduler: LRSchedulerConfig | None = None
    """
    The learning rate scheduler config for this parameter group.
    If None, the default learning rate scheduler will be used.
    """

class PretrainDatasetConfig(PretrainDatasetConfigBase, CommonDatasetConfig):
    pass


class TaskConfig(TypedConfig):
    name: str
    """Name of the task."""

    train_dataset: PretrainDatasetConfig
    """Train dataset configuration."""

    val_dataset: PretrainDatasetConfig
    """Validation dataset configuration."""

    node_energy_reduction: Literal["sum", "mean"] = "sum"
    """How to reduce the node energy scalar contributions (to get the total energy)."""

    additional_units: list[str] = []
    """Additional units to log for this task."""

    energy_loss_scale: float = 1.0
    """Scale factor for the energy loss."""
    force_loss_scale: float = 1.0
    """Scale factor for the force loss."""
    positions_loss_scale: float = 1.0
    """Scale factor for the positions loss."""
    atomic_numbers_loss_scale: float = 1.0
    """Scale factor for the atomic numbers loss."""

    # num_ctx_atoms: int | float = 1
    # """context window size for autoregressive models."""    

    normalization: dict[str, NormalizationConfig | PositionNormalizationConfig] | None = None
    """
    Normalization to apply to the target values.
    Each key is the name of the target value
    and the value is a dict with the mean and std.
    """

    cutoff: float = 12.0
    """Cutoff radius for graph generation."""
    max_neighbors: int = 30
    """Maximum number of neighbors."""


TModel = TypeVar('TModel', bound=ModelWrapperBase, infer_variance=True)

class PretrainConfig(BaseConfig, Generic[TModel]):
    model_cls: type[TModel]
    """Model class to use"""
    optimizer: OptimizerConfig
    """Optimizer to use."""
    lr_scheduler: LRSchedulerConfig | None = None
    """Learning rate scheduler configuration. If None, no learning rate scheduler is used."""

    activation: Literal[
        "scaled_silu",
        "scaled_swish",
        "silu",
        "swish",
    ] = "scaled_silu"
    """Activation function to use."""

    dropout: float | None = None
    """The dropout rate to use in GemNet."""
    edge_dropout: float | None = None
    """The percentage of edges to drop. If None, no edges are dropped."""

    embedding: EmbeddingConfig = EmbeddingConfig(
        num_elements=GOCBackboneConfig.base().num_elements,
        embedding_size=GOCBackboneConfig.base().emb_size_atom,
    )
    """Configuration for the embedding layer."""
    backbone: BackboneConfig = GOCBackboneConfig.base()
    """Configuration for the backbone."""
    output: OutputConfig = OutputConfig(num_mlps=5)
    """Configuration for the output head."""

    batch_size: int
    """Batch size to use."""
    eval_batch_size: int | None = None
    """Batch size to use for evaluation. If None, use the same as batch_size."""
    num_workers: int
    """Number of workers to use for data loading."""
    pin_memory: bool = True
    """Whether to use pin memory for data loading."""

    primary_metric: PrimaryMetricConfig
    """Primary metric to use for early stopping and checkpointing"""
    ckpt_best: CheckpointBestConfig | None = CheckpointBestConfig()
    """Configuration for saving the best checkpoint"""

    shuffle_train: bool = True
    """Should we shuffle the training dataset?"""

    shuffle_val: bool = False
    """Should we shuffle the validation dataset?"""
    
    multi_heads: bool = False
    """Whether to use multiple output heads for the model."""

    args: Namespace = Namespace()
    """Additional arguments"""

    @property
    def activation_cls(self):
        match self.activation:
            case "scaled_silu" | "scaled_swish":
                return ScaledSiLU
            case "silu" | "swish":
                return nn.SiLU
            case None:
                return nn.Identity
            case _:
                raise NotImplementedError(
                    f"Activation {self.activation} is not implemented"
                )

    log_task_losses: bool = True
    """Log the loss for each task."""
    log_task_steps_and_epochs: bool = True
    """Log the number of steps and epochs for each task."""

    tasks: list[TaskConfig]
    """List of datasets/tasks to train on."""
    mt_dataset: MTDatasetConfig = MTDatasetConfig(
        balanced=True,
        strict=True,
    )
    """Configuration for the multi-task dataset."""

    exclude_keys: list[str] = [
        "id",  # only oc20,oc22 have this
        "fid",  # only oc20,oc22 have this
        "cell_offsets",  # only oc20 has this
        "edge_index",  # only oc20 has this
        "absolute_idx",  # only ani has this
        "target_pos",  # only ani has this
        "ref_energy",  # only ani/geom have this
        "pbc",  # only ani/transition1x have this
        "oc22",  # only oc22 has this
        "name",
    ]
    """Keys to exclude when creating a batch from a data list."""

    train_on_free_atoms_only: bool = False
    """Train only on free atoms."""

    eval_on_free_atoms_only: bool = True
    """Evaluate only on free atoms."""
    
    num_ctx_atoms: int | float = 1 
    """context window size for autoregressive models."""

    energy_loss_reduction: Literal["sum", "mean"] = "mean"
    """How to reduce the energy loss. "sum" or "mean"."""
    force_loss_reduction: Literal["sum", "mean"] = "mean"
    """How to reduce the force loss. "sum" or "mean"."""
    position_loss_reduction: Literal["sum", "mean"] = "mean"
    """How to reduce the positions loss. "sum" or "mean"."""
    atomic_numbers_loss_reduction: Literal["sum", "mean"] = "mean"
    """How to reduce the atomic numbers loss. "sum" or "mean"."""

    structurewise_loss_reduction: bool = True
    """Use the proposed structurewise loss (from the paper) reduction for the force loss."""

    ema: EMAConfig | None = None
    """Configuration for the exponential moving average."""
    
    regress_forces: bool = True
    """Whether to regress forces in the model. If False, forces will not be computed 
    (as in autroegressive pretraining)."""
    
    parameter_specific_optimizers: list[ParamSpecificOptimizerConfig] | None = None
    """Configuration for parameter-specific optimizers"""

    @override
    def __post_init__(self):
        super().__post_init__()

        self.trainer.use_distributed_sampler = False
        
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
            
        # for task in self.tasks:
        #     task.num_ctx_atoms = self.num_ctx_atoms
                        

TConfig = TypeVar(
    "TConfig", bound=PretrainConfig, default=PretrainConfig, infer_variance=True
)


class _PretrainModel(LightningModuleBase[TConfig], Generic[TConfig, TModel]):
    @classmethod
    @override
    def config_cls(cls):
        return PretrainConfig[TModel]

    def metric_prefix(self) -> str:
        return f""

    def primary_metric(self, split: Literal["train", "val", "test"] | None = "val"):
        config = self.config.primary_metric
        metric = f"{config.name}"
        # if self.metric_prefix():
        #     metric = f"{self.metric_prefix()}_{config.name}"
        if split is not None:
            metric = f"{split}/{metric}"
        return metric, config.mode

    @abstractmethod
    def construct_embedding(self) -> nn.Module | None:
        pass
    
    @abstractmethod
    def construct_backbone(self) -> Backbone:
        pass
    
    @abstractmethod
    def construct_output_heads(self) -> nn.Module:
        pass

    @override
    def __init__(self, hparams: TConfig):
        super().__init__(hparams)
        self.model: TModel = self.config.model_cls(self.config)
        self.model.validate_config(hparams)
        

        # Set up callbacks
        if (ema := self.config.ema) is not None:
            self.register_callback(lambda: ema.construct_callback())

        # Set up the model
        self.embedding = self.construct_embedding()
        self.backbone = self.construct_backbone()
        self.output = self.construct_output_heads()
        
        # Set up the metrics
        self.train_metrics = FMMetrics(
            {
                task.name: {"idx": idx, "additional_units": task.additional_units}
                for idx, task in enumerate(self.config.tasks)
            },
            denormalize=any(task.normalization for task in self.config.tasks),
            free_atoms_only=self.config.eval_on_free_atoms_only,
        )
        self.val_metrics = FMMetrics(
            {
                task.name: {"idx": idx, "additional_units": task.additional_units}
                for idx, task in enumerate(self.config.tasks)
            },
            denormalize=any(task.normalization for task in self.config.tasks),
            free_atoms_only=self.config.eval_on_free_atoms_only,
        )

        # GemNet-OC re-uses some parameters at every layer.
        # We need to make sure that these parameters' gradients are
        # downscaled by the number of layers so that the gradients
        # are not too large.
        try:
            if self.backbone.shared_parameters:
                self.register_shared_parameters(self.backbone.shared_parameters)
        except AttributeError:
            log.warning(
                "The backbone does not have shared parameters. "
                "This is expected for non-GemNet-OC backbones."
            )

        self._train_dataset_sizes: list[int] | None = None
        if self.config.log_task_steps_and_epochs:
            task_steps: dict[str, SumMetric] = {}
            for task in self.config.tasks:
                metric = SumMetric()
                metric.persistent(True)
                task_steps[task.name] = metric
            self.task_steps = TypedModuleDict(task_steps)

        # Sanity check: ensure all named_parameters have requires_grad=True,
        #   otherwise add them to ignored_parameters.
        self.ignored_parameters = set[nn.Parameter]()
        for name, param in self.named_parameters():
            if param.requires_grad:
                continue
            self.ignored_parameters.add(param)
            log.info(f"Adding {name} to ignored_parameters")

        if (ckpt_best := self.config.ckpt_best) is not None:
            if (monitor := ckpt_best.monitor) is None:
                monitor, mode = self.primary_metric()
            else:
                if (mode := ckpt_best.mode) is None:
                    mode = "min"
            self.register_callback(lambda: ModelCheckpoint(monitor=monitor, mode=mode))

        self.register_callback(lambda: ModelCheckpoint(save_on_train_epoch_end=True, save_top_k=-1, filename="{epoch:02d}"))
        self.register_callback(lambda: ModelCheckpoint(save_last=True, save_top_k=0))

    def backbone_state_dict(self):
        return {
            "backbone": self.backbone.state_dict(),
            "embedding": self.embedding.atom_embedding.state_dict(),
        }

    @override
    def on_train_batch_start(self, batch: BaseData, batch_idx: int):
        if not self.config.log_task_steps_and_epochs:
            return

        assert self._train_dataset_sizes
        task_mask = batch.task_mask  # (b, t)
        task_idx = reduce(task_mask, "b t -> t", "sum")  # (t,)
        for idx, task in enumerate(self.config.tasks):
            metric = self.task_steps[task.name]
            metric(task_idx[idx])

            step = metric.compute()
            self.log(f"train/{task.name}/step", step)

            epoch = step / self._train_dataset_sizes[idx]
            self.log(f"train/{task.name}/epoch", epoch)

    @abstractmethod
    def forward(self, batch: BaseData):
        pass

    def _task_idx_onehot(self, task_idx: int):
        return F.one_hot(
            torch.tensor([task_idx], device=self.device, dtype=torch.long),
            num_classes=len(self.config.tasks),
        ).bool()

    def _force_loss(
        self, batch: BaseData, forces: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.debug:
            assert forces.shape == batch.force.shape

        pred: torch.Tensor = rearrange(forces, "n p t -> n t p")
        target: torch.Tensor = rearrange(batch.force, "n p t -> n t p")

        mask = batch.task_mask  # b t
        mask = mask[batch.batch]  # n t
        if self.config.train_on_free_atoms_only:
            mask = mask & rearrange(~batch.fixed, "n -> n 1")

        force_loss = F.pairwise_distance(pred, target, p=2.0)  # (n, t)

        if (scale := getattr(batch, "force_scale", None)) is not None:
            # force_loss_scale: (b,)
            scale = scale[batch.batch]  # (n, t)
            if self.config.train_on_free_atoms_only:
                scale = scale[~batch.fixed]
            force_loss = force_loss * scale

        if (scale := getattr(batch, "force_scale_node", None)) is not None:
            # force_scale_node: (n, t)
            if self.config.train_on_free_atoms_only:
                scale = scale[~batch.fixed]
            force_loss = force_loss * scale

        force_loss = force_loss.masked_fill(~mask, 0.0)

        if self.config.log_task_losses:
            with torch.no_grad():
                for task_idx, task in enumerate(self.config.tasks):
                    task_mask = mask & self._task_idx_onehot(task_idx)
                    task_force_loss = force_loss.masked_fill(~task_mask, 0.0)
                    self.log(
                        f"{task.name}/force_loss",
                        self._reduce_loss(
                            task_force_loss,
                            task_mask,
                            reduction=self.config.force_loss_reduction,
                        ),
                    )

        # force_loss = self._reduce_force_loss(force_loss, mask)
        return force_loss, mask

    def _energy_loss(
        self, batch: BaseData, energy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = batch.task_mask  # (b, h)

        energy_loss = F.l1_loss(energy, batch.y, reduction="none")  # (b, num_tasks)

        if (scale := getattr(batch, "y_scale", None)) is not None:
            energy_loss = energy_loss * scale  # (b, t)

        energy_loss = energy_loss.masked_fill(~mask, 0.0)

        if self.config.log_task_losses:
            with torch.no_grad():
                for task_idx, task in enumerate(self.config.tasks):
                    task_mask = mask & self._task_idx_onehot(task_idx)
                    task_energy_loss = energy_loss.masked_fill(~task_mask, 0.0)
                    self.log(
                        f"{task.name}/energy_loss",
                        self._reduce_loss(
                            task_energy_loss,
                            task_mask,
                            reduction=self.config.energy_loss_reduction,
                        ),
                    )

        return energy_loss, mask

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

    def compute_losses(
        self, batch: BaseData, energy: torch.Tensor, forces: torch.Tensor
    ):
        # Compute the energy loss
        energy_loss, energy_loss_mask = self._energy_loss(
            batch, energy
        )  # (b, t), (b, t)
        energy_loss = self._reduce_loss(
            energy_loss, energy_loss_mask, reduction=self.config.energy_loss_reduction
        )
        self.log("energy_loss", energy_loss)

        # Compute the force loss
        force_loss, force_loss_mask = self._force_loss(batch, forces)
        if self.config.structurewise_loss_reduction:
            # Compute the per-structure force loss
            force_loss = scatter(force_loss, batch.batch, dim=0, reduce="sum")  # (b, t)
            force_loss_mask_natoms = scatter(
                force_loss_mask.float(), batch.batch, dim=0, reduce="sum"
            )  # (b, t)
            force_loss = self._safe_divide(force_loss, force_loss_mask_natoms)  # (b, t)
            force_loss_mask = force_loss_mask_natoms > 0.0  # (b, t)
        force_loss = self._reduce_loss(
            force_loss, force_loss_mask, reduction=self.config.force_loss_reduction
        )
        self.log("force_loss", force_loss)

        # Combine the losses
        loss = energy_loss + force_loss
        self.log("loss", loss)

        return loss

    @override
    def training_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(
                prefix="train/",
                # batch_size=self.config.batch_size,
            ):
            energy, forces = self(batch)

            loss = self.compute_losses(batch, energy=energy, forces=forces)
            self.log_dict(self.train_metrics(batch, energy=energy, forces=forces))

            torch.cuda.empty_cache()

            return loss

    @override
    def validation_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(
                prefix="val/", 
                # batch_size=self.config.eval_batch_size 
            ):
            energy, forces = self(batch)

            loss = self.compute_losses(batch, energy=energy, forces=forces)
            metrics = self.val_metrics(batch, energy=energy, forces=forces)
            self.log_dict(metrics)


    def named_parameters_matching_patterns(self, patterns: list[str]):
        for name, param in self.named_parameters():
            if param in self.ignored_parameters:
                continue
            if (
                matching_pattern := next(
                    (pattern for pattern in patterns if fnmatch.fnmatch(name, pattern)),
                    None,
                )
            ) is None:
                continue

            yield name, param, matching_pattern
            
    def split_parameters(self, pattern_lists: list[list[str]]):
        all_parameters = list(self.parameters())

        parameters: list[list[torch.nn.Parameter]] = []
        for patterns in pattern_lists:
            matching = [
                p for _, p, _ in self.named_parameters_matching_patterns(patterns)
            ]
            parameters.append(matching)
            # remove matching parameters from all_parameters
            all_parameters = [
                p for p in all_parameters if all(p is not m for m in matching)
            ]

        return parameters, all_parameters
            
    def _cos_annealing_hparams(
        self, lr_config: WarmupCosineLambdaLRSchedulerConfig, *, lr_initial: float
    ):
        if (warmup_steps := lr_config.warmup_steps) is None:
            if warmup_epochs := lr_config.warmup_epochs:
                assert warmup_epochs >= 0, f"Invalid warmup_epochs: {warmup_epochs}"
                _ = self.trainer.estimated_stepping_batches  # make sure dataloaders are loaded for self.trainer.num_training_batches
                num_steps_per_epoch = math.ceil(
                    self.trainer.num_training_batches
                    / self.trainer.accumulate_grad_batches
                )
                warmup_steps = warmup_epochs * num_steps_per_epoch
            else:
                warmup_steps = 0
        log.critical(f"Computed warmup_steps: {warmup_steps}")

        if not (max_steps := lr_config.max_steps):
            if max_epochs := lr_config.max_epochs:
                _ = self.trainer.estimated_stepping_batches  # make sure dataloaders are loaded for self.trainer.num_training_batches
                num_steps_per_epoch = math.ceil(
                    self.trainer.num_training_batches
                    / self.trainer.accumulate_grad_batches
                )
                max_steps = max_epochs * num_steps_per_epoch
            else:
                max_steps = self.trainer.estimated_stepping_batches
                assert math.isfinite(max_steps), f"{max_steps=} is not finite"
                max_steps = int(max_steps)

        log.critical(f"Computed max_steps: {max_steps}")

        assert (
            lr_config.lr_min_factor > 0 and lr_config.lr_min_factor <= 1
        ), f"Invalid {lr_config.lr_min_factor=}"
        min_lr = lr_initial * lr_config.lr_min_factor

        lr_scheduler_hparams = dict(
            warmup_epochs=warmup_steps,
            warmup_factor=lr_config.warmup_factor,
            max_epochs=max_steps,
            lr_min_factor=lr_config.lr_min_factor,
        )
        
        log.critical(
            f"\nConstructed lr_scheduler_hparams: {lr_scheduler_hparams}\n"
        )

        return lr_scheduler_hparams

    def configure_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> LRSchedulerConfigType | None:
        match self.config.lr_scheduler:
            case None:
                return None
            case (
                WarmupCosineSchedulerConfig() | WarmupCosineLambdaLRSchedulerConfig() 
                  as config
                  ):
                raise Exception(
                    f"{type(config)} configures"
                    "PerParamGroupWarmupCosineLR scheduler."
                )
            case LinearWarmupCosineAnnealingSchedulerConfig() as config:
                if not (max_steps := config.max_steps):
                    if max_epochs := config.max_epochs:
                        _ = self.trainer.estimated_stepping_batches  # make sure dataloaders are loaded for self.trainer.num_training_batches
                        num_steps_per_epoch = math.ceil(
                            self.trainer.num_training_batches
                            / self.trainer.accumulate_grad_batches
                        )
                        max_steps = max_epochs * num_steps_per_epoch
                    else:
                        max_steps = self.trainer.estimated_stepping_batches
                        assert math.isfinite(max_steps), f"{max_steps=} is not finite"
                        max_steps = int(max_steps)

                    log.critical(f"Setting {max_steps=} by default.")

                optim_lr = float(optimizer.param_groups[0]["lr"])
                min_lr = optim_lr * config.min_lr_factor
                warmup_start_lr = optim_lr * config.warmup_start_lr_factor
                lr_scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=config.warmup_steps,
                    max_epochs=max_steps,
                    warmup_start_lr=warmup_start_lr,
                    eta_min=min_lr,
                    last_epoch=config.last_step,
                )
                return {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "strict": True,  # type: ignore
                }

            case _:
                assert_never(self.config.lr_scheduler)
                
    def configure_optimizers_param_specific_optimizers(
        self, configs: list[ParamSpecificOptimizerConfig]
    ):
        params_list, rest_params = self.split_parameters(
            [c.paremeter_patterns for c in configs]
        )
        optimizer = optimizer_from_config(
            [
                *(
                    (
                        self.config.optimizer if c.optimizer is None else c.optimizer,
                        params,
                        c.name or ",".join(c.paremeter_patterns),
                    )
                    for c, params in zip(configs, params_list)
                ),
                (self.config.optimizer, rest_params, "rest"),
            ],
            base=self.config.optimizer,
        )

        out: dict[str, Any] = {
            "optimizer": optimizer,
        }
        if (lr_config := self.config.lr_scheduler) is None:
            return out

        match lr_config:
            case LinearWarmupCosineAnnealingSchedulerConfig():
                # assert all(
                #     c.lr_scheduler is None for c in configs
                # ), f"lr_scheduler is not None for some configs: {configs=}"

                # if (
                #     lr_scheduler := self._construct_lr_scheduler(optimizer, lr_config)
                # ) is not None:
                #     out["lr_scheduler"] = lr_scheduler
                raise Exception(
                    "LinearWarmupCosineAnnealingSchedulerConfig not"
                    "supported for param-specific optimizers."
                )
            case WarmupCosineSchedulerConfig():
                raise Exception(
                    "WarmupCosineSchedulerConfig can be supported "
                    "through PerParamGroupWarmupCosineLR, but"
                    " it is not implemented."
                )
            case WarmupCosineLambdaLRSchedulerConfig():
                param_group_lr_scheduler_settings = [
                    *(
                        self._cos_annealing_hparams(
                            (
                                lr_config
                                if c.lr_scheduler is None
                                or not isinstance(c.lr_scheduler, WarmupCosineLambdaLRSchedulerConfig)
                                else c.lr_scheduler
                            ),
                            lr_initial=param_group["lr"],
                        )
                        for c, param_group in zip(configs, optimizer.param_groups[:-1])
                    ),
                    self._cos_annealing_hparams(
                        lr_config, lr_initial=optimizer.param_groups[-1]["lr"]
                    ),
                ]

                log.critical(f"{param_group_lr_scheduler_settings=}")
                # lr_scheduler = PerParamGroupLinearWarmupCosineAnnealingLR(
                #     optimizer,
                #     param_group_lr_scheduler_settings,
                # )
                lr_scheduler = LambdaLR(
                    optimizer,
                    lr_lambda=[
                        lambda step: self._cosine_lr_lambda(
                            step,
                            settings,
                        ) for settings in param_group_lr_scheduler_settings
                    ],
                )
                out["lr_scheduler"] = {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            case _:
                assert_never(lr_config)

        return out
    
    def _cosine_lr_lambda(self, current_step, scheduler_params):
        warmup_epochs = scheduler_params['warmup_epochs']
        lr_warmup_factor = scheduler_params['warmup_factor']
        max_epochs = scheduler_params['max_epochs']
        lr_min_factor = scheduler_params['lr_min_factor']
        
        # `warmup_epochs` is already multiplied with the num of iterations
        if current_step <= warmup_epochs:
            alpha = current_step / float(warmup_epochs)
            return lr_warmup_factor * (1.0 - alpha) + alpha
        else:
            if current_step >= max_epochs:
                return lr_min_factor
            lr_scale = lr_min_factor + 0.5 * (1 - lr_min_factor) * (
                1 + math.cos(math.pi * (current_step / max_epochs)))
            return lr_scale

    @override
    def configure_optimizers(self):
        if self.config.parameter_specific_optimizers is not None:
            return self.configure_optimizers_param_specific_optimizers(
                self.config.parameter_specific_optimizers
            )
            
        optimizer = optimizer_from_config([(self.config.optimizer, self.parameters())])
        out: OptimizerLRSchedulerConfig = {"optimizer": optimizer}
        if (lr_scheduler := self.configure_lr_scheduler(optimizer)) is not None:
            out["lr_scheduler"] = lr_scheduler

        return out




    def _task_dataset(self, task: TaskConfig, training: bool):
        config = task.val_dataset if not training else task.train_dataset
        dataset = PretrainLmdbDataset(config)
        dataset = wrap_common_dataset(dataset, config)
        all_pos = None
        if self.config.args.position_norm and training:
            all_pos = DT.set_task_normalization_config(dataset, task)

        # # Apply data transform to the dataset
        # if (transform := getattr(self, f"{task.name}_transform")) is None:
        #     raise ValueError(f"Transform not defined for {task.name}")
        # transform = cast(
        #     Callable[[BaseData], BaseData], partial(transform, training=training)
        # )

        # # Apply normalization to the dataset
        # if task.normalization:
        #     log.info(f"Normalizing {task.name} with {task.normalization}")
        #     transform = T.compose([transform, T.normalize(task.normalization)])

        # dataset = DT.transform(dataset, transform)

        return dataset, all_pos

    def _construct_fm_datasets(self, training: bool):

        # Construct datasets for each task
        datasets, all_pos = [], []
        for task in self.config.tasks:
            dataset, all_pos_task = self._task_dataset(task, training=training)
            datasets.append(dataset)
            all_pos.append(all_pos_task)

        # Compute the global standard deviation of the positions.
        if self.config.args.position_norm and training:
            all_pos = torch.concat(all_pos, dim=0) if all_pos else None
            scalar_std = all_pos.norm(dim=1).std()

        # Update normalization configuration for each task and execute transforms.
        for idx, task in enumerate(self.config.tasks):
            if self.config.args.position_norm and training:
                task.normalization.update({
                    "pos": PositionNormalizationConfig(
                        mean=0,
                        std=float(scalar_std),
                    )
                })
                
                # print("Updated normalization configuration after filtering:", 
                #     task.normalization) 

            # Apply data transform to the dataset
            if (transform := getattr(self, f"{task.name}_transform")) is None:
                raise ValueError(f"Transform not defined for {task.name}")
            transform = cast(
                Callable[[BaseData], BaseData], partial(transform, training=training)
            )

            # Apply normalization to the dataset
            if task.normalization:
                log.info(f"Normalizing {task.name} with {task.normalization}")
                transform = T.compose([transform, T.normalize(task.normalization)])
                
            datasets[idx] = DT.transform(datasets[idx], transform)
            

        return datasets

    @cache
    def train_dataset(self):
        datasets = self._construct_fm_datasets(training=True)
        if self.config.args.enable_mol_shuffle:
            num_perms_train = self.config.args.num_perms_train
            datasets = [PermutedDataset(d, num_perms=num_perms_train) for d in datasets]
        self._train_dataset_sizes = [len(d) for d in datasets]
        # if self.config.log_task_steps_and_epochs:
        dataset = MTSampledDataset(
            datasets,
            self.config.mt_dataset,
            ignore_balancing=False,
            num_tasks=len(self.config.tasks),
        )
        dataset = DT.transform(dataset, self.train_data_transform)
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
        datasets = self._construct_fm_datasets(training=False)
        if self.config.args.enable_mol_shuffle_eval:
            num_perms_val = self.config.args.num_perms_val
            datasets = [PermutedDataset(d, num_perms=num_perms_val) for d in datasets]
        dataset = MTSampledDataset(
            datasets,
            self.config.mt_dataset,
            ignore_balancing=True,
            num_tasks=len(self.config.tasks),
        )
        dataset = DT.transform(dataset, self.val_data_transform)
        return dataset

    def collate_fn(self, data_list: list[BaseData]):
        return Batch.from_data_list(data_list, exclude_keys=self.config.exclude_keys)

    def distributed_sampler(self, dataset: Dataset, shuffle: bool, mol_shuffle_training: bool = False):
        if mol_shuffle_training:
            return EpochAwareDistributedSampler(
                dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=shuffle,
            )
        else:
            return DistributedSampler(
                dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=shuffle,
            )

    @override
    def train_dataloader(self):
        dataset = self.train_dataset()
        sampler = self.distributed_sampler(dataset, shuffle=self.config.shuffle_train, mol_shuffle_training=self.config.args.enable_mol_shuffle)
        batch_sampler = BalancedBatchSampler(
            sampler,
            batch_size=self.config.batch_size,
            device=self.device,
        )
        data_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        return data_loader

    @override
    def val_dataloader(self):
        dataset = self.val_dataset()
        sampler = self.distributed_sampler(dataset, shuffle=self.config.shuffle_val)
        batch_sampler = BalancedBatchSampler(
            sampler,
            batch_size=self.config.eval_batch_size,
            device=self.device,
        )
        data_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        return data_loader

    def _task_config(self, name: str):
        return next((task for task in self.config.tasks if task.name == name), None)

    @staticmethod
    def _to_int(value):
        return int(value.item() if torch.is_tensor(value) else value)

    def train_data_transform(self, data: BaseData):
        data = self.data_transform(data)
        return data

    def val_data_transform(self, data: BaseData):
        data = self.data_transform(data)
        return data

    def data_transform(self, data: BaseData):
        data.y = (
            data.y.float()
            if torch.is_tensor(data.y)
            else torch.tensor(data.y, dtype=torch.float)
        )

        data.fixed = data.fixed.bool()
        data.atomic_numbers = data.atomic_numbers.long()
        data.natoms = self._to_int(data.natoms)
        data.sid = self._to_int(data.sid)
        for graph_type in ["main", "a2a", "a2ee2a", "qint"]:
            key = f"{graph_type}_num_neighbors"
            setattr(data, key, self._to_int(data[key]))

        for attr in ("y", "force"):
            key = f"{attr}_scale"
            if not hasattr(data, key):
                raise ValueError(f"{key=} not found in data")

        # make all tensors contiguous
        for key in data.keys():
            if not torch.is_tensor(data[key]):
                continue

            data[key] = data[key].contiguous()

        return data

    def _process_aint_graph(self, graph: Graph, *, training: bool):
        if self.config.edge_dropout:
            graph["edge_index"], mask = dropout_edge(
                graph["edge_index"],
                p=self.config.edge_dropout,
                training=training,
            )
            graph["distance"] = graph["distance"][mask]
            graph["vector"] = graph["vector"][mask]
            graph["cell_offset"] = graph["cell_offset"][mask]

            if "id_swap_edge_index" in graph:
                graph["id_swap_edge_index"] = graph["id_swap_edge_index"][mask]

        return graph

    def _generate_graphs(
        self,
        data: BaseData,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        pbc: bool,
        *,
        training: bool,
    ):
        if self.config.model_cls == models.GemNetModelWrapper:
            normalize_edge_vector = True
        else:
            normalize_edge_vector = False

        aint_graph = generate_graph(
            data, cutoff=cutoffs.aint, max_neighbors=max_neighbors.aint, 
            pbc=pbc, normalize_edge_vector=normalize_edge_vector
        )
        aint_graph = self._process_aint_graph(aint_graph, training=training)
        subselect = partial(
            subselect_graph,
            data,
            aint_graph,
            cutoff_orig=cutoffs.aint,
            max_neighbors_orig=max_neighbors.aint,
        )
        main_graph = subselect(cutoffs.main, max_neighbors.main)
        aeaint_graph = subselect(cutoffs.aeaint, max_neighbors.aeaint)
        qint_graph = subselect(cutoffs.qint, max_neighbors.qint)

        # We can't do this at the data level: This is because the batch collate_fn doesn't know
        # that it needs to increment the "id_swap" indices as it collates the data.
        # So we do this at the graph level (which is done in the GemNetOC `get_graphs_and_indices` method).
        # main_graph = symmetrize_edges(main_graph, num_atoms=data.pos.shape[0])
        
        if hasattr(self.config.backbone, "qint_tags"):
            tags = self.config.backbone.qint_tags
        else:
            tags = [0, 1, 2]
        qint_graph = tag_mask(data, qint_graph, tags=tags)
        
        graphs = {
            "main": main_graph,
            "a2a": aint_graph,
            "a2ee2a": aeaint_graph,
            "qint": qint_graph,
        }
        
        # try:
        #     qint_graph = tag_mask(data, qint_graph, tags=self.config.backbone.qint_tags)

        #     graphs = {
        #         "main": main_graph,
        #         "a2a": aint_graph,
        #         "a2ee2a": aeaint_graph,
        #         "qint": qint_graph,
        #     }
        # except AttributeError:
        #     graphs = {
        #         "main": main_graph,
        #         "a2a": aint_graph,
        #         "a2ee2a": aeaint_graph,
        #     }

        for graph_type, graph in graphs.items():
            graph["num_neighbors"] = graph["edge_index"].shape[1]
            for key, value in graph.items():
                setattr(data, f"{graph_type}_{key}", value)

        return data

    def _initial_data_transform(self, data: BaseData):
        if not torch.is_tensor(data.y):
            data.y = torch.tensor(data.y)
        data.y = data.y.view(-1)

        return data

    def oc20_transform(self, data: BaseData, *, training: bool):
        if hasattr(data, "pos_relaxed"):
            # Use the relaxed position as a target for our autoregressive objective
            data.pos = data.pos_relaxed
            data.y = torch.tensor(data.y_init)
        
        data = self._initial_data_transform(data)
        assert (
            config := self._task_config("oc20")
        ) is not None, "OC20 task is not configured"

        # convert back these keys into required format for collation
        data.natoms = int(data.natoms.item() if torch.is_tensor(data) else data.natoms)

        data.atomic_numbers = data.atomic_numbers.long()
        data.tags = data.tags.long()
        
        data = self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(
                config.cutoff / getattr(
                    config.normalization, "pos", {"std": 1.0}
                    )["std"]
                ),
            # max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            # 20 for OC20 EqV2
            max_neighbors=MaxNeighbors.from_goc_base_proportions(
                config.max_neighbors
            ),
            pbc=False if self.config.args.no_pbc else True,
            training=training,
        )
        
        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale
        data.positions_scale = config.positions_loss_scale
        data.atomic_numbers_scale = config.atomic_numbers_loss_scale
            
        return data

    def oc22_transform(self, data: BaseData, *, training: bool):
        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("oc22")
        ) is not None, "OC22 task is not configured"

        # convert back these keys into required format for collation
        data.natoms = int(data.natoms.item() if torch.is_tensor(data) else data.natoms)

        data.atomic_numbers = data.atomic_numbers.long()
        data.tags = data.tags.long()
        try:
            data.y = torch.tensor(float(data.y)).view(-1)
        except BaseException:
            data.y = torch.tensor(float(data.y_relaxed)).view(-1)
        data.name = "oc22"

        data = self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(
                config.cutoff / getattr(
                    config.normalization, "pos", {"std": 1.0}
                    )["std"]
                ),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(
                config.max_neighbors
            ),
            pbc=False if self.config.args.no_pbc else True,
            training=training,
        )

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale
        data.positions_scale = config.positions_loss_scale
        data.atomic_numbers_scale = config.atomic_numbers_loss_scale

        return data
    
    def odac_transform(self, data: BaseData, *, training: bool):
        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("odac")
        ) is not None, "ODAC task is not configured"

        # convert back these keys into required format for collation
        data.natoms = int(data.natoms.item() if torch.is_tensor(data) else data.natoms)

        data.atomic_numbers = data.atomic_numbers.long()
        data.tags = data.tags.long()

        data = self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(
                config.cutoff / getattr(
                    config.normalization, "pos", {"std": 1.0}
                    )["std"]
                ),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(
                config.max_neighbors
            ),
            pbc=False if self.config.args.no_pbc else True,
            training=training,
        )

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale
        data.positions_scale = config.positions_loss_scale
        data.atomic_numbers_scale = config.atomic_numbers_loss_scale

        return data

    @staticmethod
    def _set_inf_cell(data: BaseData, max_length: float = 1000.0):
        data.cell = (torch.eye(3) * max_length).unsqueeze(dim=0)
        return data

    def ani1x_transform(self, data: BaseData, *, training: bool):
        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("ani1x")
        ) is not None, "ANI1x task is not configured"

        data.y = data.y.view(-1).float()
        if not hasattr(data, "sid"):
            data.sid = data.absolute_idx
        if not hasattr(data, "natoms"):
            data.natoms = data.num_nodes

        # data.fixed = torch.ones(data.natoms)
        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()
        data.name = "ani1x"

        data = self._set_inf_cell(data)
        data = self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(
                config.cutoff / getattr(
                    config.normalization, "pos", {"std": 1.0}
                    )["std"]
                ),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(
                config.max_neighbors
            ),
            pbc=False,
            training=training,
        )

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale
        data.positions_scale = config.positions_loss_scale
        data.atomic_numbers_scale = config.atomic_numbers_loss_scale
        
        return data

    def transition1x_transform(self, data: BaseData, *, training: bool):
        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("transition1x")
        ) is not None, "Transition1x task is not configured"

        data.y = data.y.view(-1).float()
        if not hasattr(data, "sid"):
            data.sid = data.absolute_idx
        if not hasattr(data, "natoms"):
            data.natoms = data.num_nodes

        # data.fixed = torch.ones(data.natoms)
        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()
        data.name = "transition1x"

        data = self._set_inf_cell(data)
        data = self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(
                config.cutoff / getattr(
                    config.normalization, "pos", {"std": 1.0}
                    )["std"]
                ),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(
                config.max_neighbors
            ),
            pbc=False,
            training=training,
        )

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale
        data.positions_scale = config.positions_loss_scale
        data.atomic_numbers_scale = config.atomic_numbers_loss_scale
        
        return data


class PretrainModel(_PretrainModel[TConfig, TModel], Generic[TConfig, TModel]):
    
    @override
    def construct_embedding(self) -> nn.Module | None:
        match self.config.model_cls:
            case models.GemNetModelWrapper:
                return self.model.Embedding(self.config)
                
            case models.SchNetModelWrapper:
                return nn.Embedding(
                    self.config.embedding.num_elements, 
                    self.config.embedding.embedding_size,
                    padding_idx=0
                )
            case (models.EquiformerV2ModelWrapper | models.EScAIPModelWrapper | 
                  models.AutoregLlamaEquiformerV2ModelWrapper):
                # Handled by the backbone 
                return 
            case _:
                raise ValueError(f"Invalid model_cls: {self.config.model_cls}")
    
    @override
    def construct_backbone(self) -> Backbone:
        return self.model.BackboneType(
            cast(self.model.BackboneConfigType, self.config.backbone), # type: ignore
            **dict(self.config.backbone)
            )
        
    @override
    def construct_output_heads(self) -> nn.Module:
        return self.model.PretrainOutputHead(self.config, self.backbone)
            
    @override
    def forward(self, batch: BaseData):
        
        
        match self.config.model_cls:
            
            case models.GemNetModelWrapper:
                h = self.embedding(batch)  # (N, d_model) # type: ignore
                out = cast(GOCBackboneOutput, self.backbone(batch, h=h))
            
            # case models.SchNetModelWrapper:
            #     out = cast(SchNetBackboneOutput, self.backbone(batch))

            # case models.EScAIPModelWrapper:
            #     out = cast(EScAIPBackboneOutput, self.backbone(batch))

            case (
                    models.EquiformerV2ModelWrapper | models.AutoregLlamaEquiformerV2ModelWrapper |
                    models.AutoregressiveEquiformerV2ModelWrapper
                ):
                out = cast(EquiformerV2BackboneOutput, self.backbone(batch))
            
            case _:
                raise ValueError(f"Invalid model_cls: {self.config.model_cls}")
            
        
                
        return self.output(batch, out)  # (n h), (n p h)
    
    
    
class AutoregressivePretrainModel(
    PretrainModel[TConfig, models.AutoregressiveEquiformerV2ModelWrapper], Generic[TConfig]
    ):
    
    @override
    def __init__(self, hparams: TConfig):
        super().__init__(hparams)
        
        # TODO: Reflect autoregressive model metrics
        # Set up the metrics
        self.train_metrics = FMMetrics(
            {
                task.name: {
                    "idx": idx, "additional_units": task.additional_units, 
                    "num_ctx_atoms": self.config.num_ctx_atoms
                    }
                for idx, task in enumerate(self.config.tasks)
            },
            denormalize=any(task.normalization for task in self.config.tasks),
            free_atoms_only=self.config.eval_on_free_atoms_only,
            num_classes=self.backbone.max_num_elements
        )
        self.val_metrics = FMMetrics(
            {
                task.name: {
                    "idx": idx, "additional_units": task.additional_units,
                    "num_ctx_atoms": self.config.num_ctx_atoms
                    }
                for idx, task in enumerate(self.config.tasks)
            },
            denormalize=any(task.normalization for task in self.config.tasks),
            free_atoms_only=self.config.eval_on_free_atoms_only,
            num_classes=self.backbone.max_num_elements
        )
    
    @override
    def construct_embedding(self) -> nn.Module | None:
        return None  # Handled by the backbone
    
    @override
    def construct_backbone(self) -> Backbone:
        return self.model.BackboneType(
            cast(self.model.BackboneConfigType, self.config.backbone), # type: ignore
            **dict(self.config.backbone)
            )
        
    @override
    def construct_output_heads(self) -> nn.Module:
        return self.model.PretrainOutputHead(self.config, self.backbone)
            
    @override
    def forward(self, batch: BaseData):
        
        
        match self.config.model_cls:
            
            case (
                    models.AutoregressiveEquiformerV2ModelWrapper
                ):
                out = cast(EquiformerV2BackboneOutput, self.backbone(batch))
            
            case _:
                raise ValueError(f"Invalid (autoregressive) model_cls: {self.config.model_cls}")
            
        return self.output(batch, out)  # (n h), (n p h)
    
    
    def _position_loss(self, batch, positions):
        if self.debug:
            assert positions.shape == batch.pos.shape
        
        pred   = rearrange(positions, "n p t -> n t p")
        target = rearrange(batch.pos,    "n p t -> n t p")

        mask = batch.task_mask[batch.batch]
        if self.config.train_on_free_atoms_only:
            mask = mask & rearrange(~batch.fixed, "n -> n 1")

        # build context window mask over atoms → (n,) 
        N = batch.pos.size(0)
        ctx_mask = torch.zeros(N, dtype=torch.bool, device=pred.device)
        num_ctx = self.config.num_ctx_atoms
        for i in range(batch.num_graphs):
            start, end = int(batch.ptr[i]), int(batch.ptr[i+1])
            skip = max(1, int(num_ctx if num_ctx >= 1 else num_ctx * (end-start)))
            if end - start > skip:
                ctx_mask[start+skip - 1: end - 1] = True
                # ctx_mask[start+skip: end - 1] = True

        mask &= rearrange(ctx_mask, "n -> n 1")  # (n, t)

        # build shifted target & mask out final atom of each graph 
        # target_shifted[j] = true pos of atom j+1
        target_shifted = torch.zeros_like(pred)
        for i in range(N-1):
            # if ctx_mask[i] and ctx_mask[i+1]:
            if ctx_mask[i]:
                target_shifted[i] = target[i+1]

        _d = F.l1_loss(pred, target_shifted, reduction="none")  # (n, t, 3)\
        _d = _d.mean(-1)                                           # (n, t)

        pos_loss = _d.masked_fill(~mask, 0.0)
        if (ps := getattr(batch, "positions_scale", None)) is not None:
            scale = ps[batch.batch]
            if self.config.train_on_free_atoms_only:
                scale = scale[~batch.fixed]
            pos_loss = pos_loss * scale

        if self.config.log_task_losses:
            with torch.no_grad():
                for task_idx, task in enumerate(self.config.tasks):
                    task_mask = mask & self._task_idx_onehot(task_idx)
                    task_pos_loss = pos_loss.masked_fill(~task_mask, 0.0)
                    self.log(
                        f"{task.name}/positions_loss",
                        self._reduce_loss(
                            task_pos_loss,
                            task_mask,
                            reduction=self.config.force_loss_reduction,
                        ),
                    )

        # force_loss = self._reduce_force_loss(force_loss, mask)
        return pos_loss, mask
    
        
    def _atomic_number_loss(
        self,
        batch: Batch,
        atomic_numbers_pred: torch.Tensor  # (N_atoms, num_classes, num_tasks)
    ) -> tuple[torch.Tensor, torch.BoolTensor]:


        # 1) bring logits into (N, T, C) and get targets (N, T)
        pred   = rearrange(atomic_numbers_pred, "n c t -> n t c")          # (N, T, C)
        target = batch.atomic_numbers.long()                              # (N, T)

        # 2) task‐mask & optional free‐atom filter → (N, T)
        mask = batch.task_mask[batch.batch]                               # (N, T)
        if self.config.train_on_free_atoms_only:
            mask &= rearrange(~batch.fixed, "n -> n 1")

        # 3) context‐window mask over atoms → (N,)
        N = pred.size(0)
        ctx_mask = torch.zeros(N, dtype=torch.bool, device=pred.device)
        num_ctx = self.config.num_ctx_atoms
        for gi in range(batch.num_graphs):
            s, e = int(batch.ptr[gi]), int(batch.ptr[gi+1])
            sz = e - s
            skip = max(1, int(num_ctx) if num_ctx >= 1 else int(num_ctx * sz))
            if sz > skip:
                ctx_mask[s + skip - 1: e - 1] = True

        mask &= ctx_mask.unsqueeze(1)                                     # (N, T)

        # 4) build the “next‐atom” targets → (N, T)
        target_shifted = torch.zeros_like(target)
        for j in range(N - 1):
            # if ctx_mask[j] and ctx_mask[j + 1]:
            if ctx_mask[j]:
                target_shifted[j] = target[j + 1]

        # 5) flatten for CE
        N, T, C = pred.size(0), pred.size(1), pred.size(2)
        pred_flat   = pred.reshape(N * T, C)
        target_flat = target_shifted.reshape(N * T)
        mask_flat   = mask.reshape(N * T)

        # 6) compute per‐example CE (none) and scatter back → (N*T,)
        if mask_flat.any():
            ce = F.cross_entropy(
                pred_flat[mask_flat], target_flat[mask_flat], reduction="none"
            )
            loss_flat = torch.zeros(N * T, device=pred.device)
            loss_flat[mask_flat] = ce
        else:
            loss_flat = torch.zeros(N * T, device=pred.device)

        # 7) reshape back → (N, T)
        loss = loss_flat.view(N, T)

        # 8) per‐task logging
        if self.config.log_task_losses:
            with torch.no_grad():
                for idx, task in enumerate(self.config.tasks):
                    task_mask = mask & self._task_idx_onehot(idx)  # (N, T)
                    task_loss = loss.masked_fill(~task_mask, 0.0)
                    self.log(
                        f"{task.name}/atomic_number_loss",
                        self._reduce_loss(
                            task_loss,
                            task_mask,
                            reduction=self.config.atomic_numbers_loss_reduction,
                        ),
                    )

        return loss, mask

  
    
    @override
    def compute_losses(
        self, batch: BaseData, energy: torch.Tensor, forces: torch.Tensor | None, 
        positions: torch.Tensor, atomic_numbers: torch.Tensor
    ):
        assert type(positions) is torch.Tensor, f"Positions must be a tensor we got {type(positions)}"
        assert type(atomic_numbers) is torch.Tensor, "Atomic numbers must be a tensor"


        # Compute the energy loss
        # energy_loss, energy_loss_mask = self._energy_loss(
        #     batch, energy
        # )  # (b, t), (b, t)
        # energy_loss = self._reduce_loss(
        #     energy_loss, energy_loss_mask, reduction=self.config.energy_loss_reduction
        # )
        # self.log("energy_loss", energy_loss)
        # print(f"Energy loss: {energy_loss.item()}")
        
        # Compute the position loss
        position_loss, position_loss_mask = self._position_loss(batch, positions)
        if self.config.structurewise_loss_reduction:
            position_loss = scatter(
                position_loss, batch.batch, dim=0, reduce="sum"
            )
            position_loss_mask_natoms = scatter(
                position_loss_mask.float(), batch.batch, dim=0, reduce="sum"
            )
            position_loss = self._safe_divide(
                position_loss, position_loss_mask_natoms
            )
            position_loss_mask = position_loss_mask_natoms > 0.0
        position_loss = self._reduce_loss(
            position_loss, position_loss_mask, reduction=self.config.position_loss_reduction
        )
        self.log("position_loss", position_loss)
        # print(f"Position loss: {position_loss.item()}")
        
        atomic_numbers_loss, atomic_numbers_mask = self._atomic_number_loss(
            batch, atomic_numbers
        )
        if self.config.structurewise_loss_reduction:
            atomic_numbers_loss = scatter(
                atomic_numbers_loss, batch.batch, dim=0, reduce="sum"
            )
            atomic_numbers_mask_natoms = scatter(
                atomic_numbers_mask.float(), batch.batch, dim=0, reduce="sum"
            )
            atomic_numbers_loss = self._safe_divide(
                atomic_numbers_loss, atomic_numbers_mask_natoms
            )
            atomic_numbers_mask = atomic_numbers_mask_natoms > 0.0
        atomic_numbers_loss = self._reduce_loss(
            atomic_numbers_loss, atomic_numbers_mask, 
            reduction=self.config.atomic_numbers_loss_reduction
        )
        self.log("atomic_numbers_loss", atomic_numbers_loss)
        # print(f"Atomic numbers loss: {atomic_numbers_loss.item()}")
        

        if forces:
            # Compute the force loss
            force_loss, force_loss_mask = self._force_loss(batch, forces)
            if self.config.structurewise_loss_reduction:
                # Compute the per-structure force loss
                force_loss = scatter(force_loss, batch.batch, dim=0, reduce="sum")  # (b, t)
                force_loss_mask_natoms = scatter(
                    force_loss_mask.float(), batch.batch, dim=0, reduce="sum"
                )  # (b, t)
                force_loss = self._safe_divide(force_loss, force_loss_mask_natoms)  # (b, t)
                force_loss_mask = force_loss_mask_natoms > 0.0  # (b, t)
            force_loss = self._reduce_loss(
                force_loss, force_loss_mask, reduction=self.config.force_loss_reduction
            )
            self.log("force_loss", force_loss)

        loss =  position_loss + atomic_numbers_loss
        self.log("loss", loss)

        return loss

    @override
    def training_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(
                prefix="train/",
                # batch_size=self.config.batch_size,
            ):
            energy, forces, positions, atomic_numbers = self(batch)

            
            loss = self.compute_losses(
                batch, energy=energy, forces=forces,
                positions=positions, atomic_numbers=atomic_numbers
            )
            
            self.log_dict(
                self.train_metrics(
                    batch, energy=energy, forces=forces,
                    positions=positions, atomic_numbers=atomic_numbers
                    )
                )

            torch.cuda.empty_cache()

            return loss

    @override
    def validation_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(
                prefix="val/", 
                # batch_size=self.config.eval_batch_size 
            ):

            energy, forces, positions, atomic_numbers = self(batch)

            loss = self.compute_losses(
                batch, energy=energy, forces=forces,
                positions=positions, atomic_numbers=atomic_numbers
            )
            
            self.log_dict(
                self.val_metrics(
                    batch, energy=energy, forces=forces,
                    positions=positions, atomic_numbers=atomic_numbers
                )
            )
