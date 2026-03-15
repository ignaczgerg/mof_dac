"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import Counter
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Callable, Protocol, cast, runtime_checkable

import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.classification
from frozendict import frozendict
from jmp.lightning import TypedConfig
from jmp.lightning.util.typed import TypedModuleDict
from torch_geometric.data.data import BaseData
from typing_extensions import override

# if TYPE_CHECKING:
#     from .base import (
#         BinaryClassificationTargetConfig,
#         MulticlassClassificationTargetConfig,
#     )

# from jmp.tasks.finetune.base import TaskConfig
from jmp.tasks.finetune.model_wrapper_base import (
    BinaryClassificationTargetConfig,
    MulticlassClassificationTargetConfig,
)


log = getLogger(__name__)

def _safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))

def symlog(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + eps)

class CheckConflictingStructuresConfig(TypedConfig):
    structures: dict[str, str] = {}
    """
    A dictionary which maps from pre-training dataset names to the path of the
    pickle files (saved using `torch.save` as sets of frozendict[int, int] objects)
    containing the unique atomic numbers in the dataset.

    The frozendict[int, int] objects are mappings from atomic numbers to the number
    of atoms with that atomic number in the structure.
    """

    all: bool = True
    """Also check for conflicting structures across all datasets (i.e., the union of all structures)"""


class MetricsConfig(TypedConfig):
    check_conflicting_structures: CheckConflictingStructuresConfig | None = None
    """
    Configuration for checking conflicting structures.

    This is used to, for example, check to see what percentage
    of structures in the fine-tuning dataset (e.g., in QM9) exist
    in the pre-training dataset (e.g., ANI-1x).
    """

    report_log_metrics: bool = False
    """Report metrics in the log-space"""

    report_normal_metrics: bool = True
    """report metrics in the normal space"""

    report_rmse: bool = False
    """Whether to report RMSE in addition to MAE"""

    report_mape: bool = False
    """Whether to report MAPE in addition to MAE"""

    report_smape: bool = False
    """Whether to report SMAPE in addition to MAE"""

    report_roi_smape: bool = False
    """Whether to report ROI SMAPE in addition to MAE"""

    report_mse: bool = False
    """Whether to report MSE in addition to MAE"""


@dataclass(frozen=True)
class MetricPair:
    predicted: torch.Tensor
    ground_truth: torch.Tensor


@runtime_checkable
class MetricPairProvider(Protocol):
    def __call__(
        self, prop: str, batch: BaseData, preds: dict[str, torch.Tensor]
    ) -> MetricPair | None: ...

@runtime_checkable
class MTMetricPairProvider(Protocol):
    def __call__(
        self, prop: str, batch: BaseData, preds: dict[str, torch.Tensor], task_idx: int
    ) -> MetricPair | None: ...

class ConflictingMetrics(nn.Module):
    @override
    def __init__(
        self,
        graph_targets: list[str],
        node_targets: list[str],
        structures: set[frozendict[int, int]],
        provider: MetricPairProvider,
    ):
        super().__init__()

        self.graph_targets = graph_targets
        self.node_targets = node_targets
        self.targets = graph_targets + node_targets

        self.structures = structures
        self.conflicting_maes = TypedModuleDict(
            {target: torchmetrics.MeanAbsoluteError() for target in self.targets}
        )
        self.non_conflicting_maes = TypedModuleDict(
            {target: torchmetrics.MeanAbsoluteError() for target in self.targets}
        )

        self.num_conflicting = torchmetrics.SumMetric()
        self.num_non_conflicting = torchmetrics.SumMetric()
        self.num_total = torchmetrics.SumMetric()

        self.provider = provider

    def _compute_mask(self, data: BaseData):
        n_graphs = int(torch.max(data.batch).item() + 1)
        mask = torch.zeros(n_graphs, dtype=torch.bool, device=data.batch.device)
        for i in range(n_graphs):
            # get the atomic numbers for the current molecule
            atomic_numbers = data.atomic_numbers[data.batch == i].long()
            atomic_numbers_dict = frozendict(Counter(atomic_numbers.tolist()))
            mask[i] = atomic_numbers_dict in self.structures
        return mask

    def _compute_metrics(
        self,
        targets: list[str],
        batch: BaseData,
        preds: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ):
        for key in targets:
            if (mp := self.provider(key, batch, preds)) is None:
                continue

            conflicting_mae = self.conflicting_maes[key]
            non_conflicting_mae = self.non_conflicting_maes[key]

            conflicting_mae(mp.predicted[mask], mp.ground_truth[mask])
            non_conflicting_mae(mp.predicted[~mask], mp.ground_truth[~mask])

    @override
    def forward(self, batch: BaseData, preds: dict[str, torch.Tensor]):
        mask = self._compute_mask(batch)

        self.num_conflicting(mask)
        self.num_non_conflicting(~mask)
        self.num_total(torch.ones_like(mask))

        self._compute_metrics(self.graph_targets, batch, preds, mask)
        self._compute_metrics(self.node_targets, batch, preds, mask[batch.batch])

        metrics: dict[str, torchmetrics.Metric] = {}
        metrics["num_conflicting"] = self.num_conflicting
        metrics["num_non_conflicting"] = self.num_non_conflicting
        metrics["num_total"] = self.num_total

        for key in self.targets:
            metrics[f"{key}_conflicting_mae"] = self.conflicting_maes[key]
            metrics[f"{key}_non_conflicting_mae"] = self.non_conflicting_maes[key]

        return metrics


class BinaryClassificationMetrics(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        assert num_classes == 2, "Only binary classification is supported"

        self.roc_auc = torchmetrics.classification.BinaryAUROC()
        self.f1 = torchmetrics.classification.F1Score(task="binary")
        self.balanced_accuracy = torchmetrics.classification.MulticlassAccuracy(
            average="macro", num_classes=2
        )

    def compute(self):
        # This method returns a Tensor which contains the metric used for RLP
        return self.balanced_accuracy.compute()

    @override
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        metrics: dict[str, torchmetrics.Metric] = {}

        self.roc_auc(pred, target)
        self.f1(pred, target)

        # For balanced accuracy, we need to convert the binary pred/target to a
        # multiclass target with 2 classes. This is because torchmetrics' implementation
        # of torchmetrics.classification.BinaryAccuracy does not support the "macro"
        # average, which is what we want.
        cls_pred = pred.new_zeros((*pred.shape, 2))
        cls_pred[..., 1] = pred
        cls_pred[..., 0] = 1 - pred

        cls_target = target.long()
        self.balanced_accuracy(cls_pred, cls_target)

        metrics["roc_auc"] = self.roc_auc
        metrics["f1"] = self.f1
        metrics["balanced_accuracy"] = self.balanced_accuracy

        return metrics


class MulticlassClassificationMetrics(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.roc_auc = torchmetrics.classification.AUROC(
            task="multiclass", num_classes=num_classes
        )
        self.f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes
        )
        self.balanced_accuracy = torchmetrics.classification.MulticlassAccuracy(
            average="macro", num_classes=num_classes
        )
        self.num_classes = num_classes

    def compute(self):
        # This method returns a Tensor which contains the metric used for RLP
        return self.balanced_accuracy.compute()

    @override
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        metrics: dict[str, torchmetrics.Metric] = {}

        self.roc_auc(pred, target)
        self.f1(pred, target)
        self.balanced_accuracy(pred, target)

        metrics["roc_auc"] = self.roc_auc
        metrics["f1"] = self.f1
        metrics["balanced_accuracy"] = self.balanced_accuracy

        return metrics


class FinetuneMetrics(nn.Module):
    @property
    def regression_targets(self):
        return (
            self.graph_scalar_targets + self.node_vector_targets 
            + [target.name for target in self.graph_classification_adabin_targets]
        )

    @override
    def __init__(
        self,
        config: MetricsConfig,
        provider: MetricPairProvider,
        cls_provider: MetricPairProvider,
        graph_scalar_targets: list[str],
        graph_classification_targets: "list[BinaryClassificationTargetConfig | MulticlassClassificationTargetConfig]",
        graph_classification_adabin_targets: "list[MulticlassClassificationTargetConfig]",
        node_vector_targets: list[str],
    ):
        super().__init__()

        if not isinstance(provider, MetricPairProvider):
            raise TypeError(
                f"Expected {provider=} to be an instance of {MetricPairProvider=}"
            )
        self.provider = provider
        self.cls_provider = cls_provider

        self.config = config
        self.graph_scalar_targets = graph_scalar_targets
        self.graph_classification_targets = graph_classification_targets
        self.graph_classification_adabin_targets = graph_classification_adabin_targets
        self.node_vector_targets = node_vector_targets

        self.maes = TypedModuleDict(
            {
                target: torchmetrics.MeanAbsoluteError()
                for target in self.regression_targets
            },
            key_prefix="mae_",
        )
        if self.config.report_normal_metrics:
            if self.config.report_rmse:
                self.rmses = TypedModuleDict(
                    {
                        target: torchmetrics.MeanSquaredError(squared=False)
                        for target in self.regression_targets
                    },
                    key_prefix="rmse_",
                )

            if self.config.report_mape:
                self.mapes = TypedModuleDict(
                    {
                        target: torchmetrics.MeanAbsolutePercentageError()
                        for target in self.regression_targets
                    },
                    key_prefix="mape_",
                )

            if self.config.report_smape:
                self.smapes = TypedModuleDict(
                    {
                        target: torchmetrics.SymmetricMeanAbsolutePercentageError()
                        for target in self.regression_targets
                    },
                    key_prefix="smape_",
                )

            if self.config.report_mse:
                self.mses = TypedModuleDict(
                    {
                        target: torchmetrics.MeanSquaredError()
                        for target in self.regression_targets
                    },
                    key_prefix="mse_",
                )

        if self.config.report_log_metrics:
            self.log_maes = TypedModuleDict(
                    {
                        target: torchmetrics.MeanAbsoluteError()
                        for target in self.regression_targets
                    },
                    key_prefix="log_mae_",
                )

            if self.config.report_rmse:
                self.log_rmses = TypedModuleDict(
                    {
                        target: torchmetrics.MeanSquaredError(squared=False)
                        for target in self.regression_targets
                    },
                    key_prefix="log_rmse_",
                )

            if self.config.report_mape:
                self.log_mapes = TypedModuleDict(
                    {
                        target: torchmetrics.MeanAbsolutePercentageError()
                        for target in self.regression_targets
                    },
                    key_prefix="log_mape_",
                )

            if self.config.report_smape:
                self.log_smapes = TypedModuleDict(
                    {
                        target: torchmetrics.SymmetricMeanAbsolutePercentageError()
                        for target in self.regression_targets
                    },
                    key_prefix="log_smape_",
                )

            if self.config.report_mse:
                self.log_mses = TypedModuleDict(
                    {
                        target: torchmetrics.MeanSquaredError()
                        for target in self.regression_targets
                    },
                    key_prefix="log_mse_",
                )

        self.cls_metrics = TypedModuleDict(
            {
                target.name: (
                    BinaryClassificationMetrics(target.num_classes)
                    if isinstance(target, BinaryClassificationTargetConfig)
                    else MulticlassClassificationMetrics(target.num_classes)
                )
                for target in (self.graph_classification_targets)
            },
            key_prefix="cls_",
        )

        if (ccs := self.config.check_conflicting_structures) is not None:
            metrics_dict: dict[str, ConflictingMetrics] = {}
            all_structures = set[frozendict[int, int]]()
            for name, structures in ccs.structures.items():
                structures = cast(set[frozendict[int, int]], torch.load(structures))
                metrics_dict[name] = ConflictingMetrics(
                    graph_targets=self.graph_scalar_targets,
                    node_targets=self.node_vector_targets,
                    structures=structures,
                    provider=self.provider,
                )
                if ccs.all:
                    all_structures.update(structures)

            if ccs.all:
                metrics_dict["all"] = ConflictingMetrics(
                    graph_targets=self.graph_scalar_targets,
                    node_targets=self.node_vector_targets,
                    structures=all_structures,
                    provider=self.provider,
                )
            self.conflicting = TypedModuleDict(metrics_dict)

    @override
    def forward(self, batch: BaseData, preds: dict[str, torch.Tensor]):
        metrics: dict[str, torchmetrics.Metric] = {}

        if self.config.report_normal_metrics:
            for key, mae in self.maes.items():
                if (mp := self.provider(key, batch, preds)) is None:
                    continue
                mae(mp.predicted, mp.ground_truth)
                metrics[f"{key}_mae"] = mae

            if self.config.report_rmse:
                for key, rmse in self.rmses.items():
                    if (mp := self.provider(key, batch, preds)) is None:
                        continue

                    rmse(mp.predicted, mp.ground_truth)
                    metrics[f"{key}_rmse"] = rmse

            if self.config.report_mape:
                for key, mape in self.mapes.items():
                    if (mp := self.provider(key, batch, preds)) is None:
                        continue

                    mape(mp.predicted, mp.ground_truth)
                    metrics[f"{key}_mape"] = mape

            if self.config.report_smape:
                for key, smape in self.smapes.items():
                    if (mp := self.provider(key, batch, preds)) is None:
                        continue

                    smape(mp.predicted, mp.ground_truth)
                    metrics[f"{key}_smape"] = smape
            
            if self.config.report_mse:
                for key, mse in self.mses.items():
                    if (mp := self.provider(key, batch, preds)) is None:
                        continue

                    mse(mp.predicted, mp.ground_truth)
                    metrics[f"{key}_mse"] = mse

        if self.config.report_log_metrics:
            for key in self.regression_targets:
                if (mp := self.provider(key, batch, preds)) is None:
                    continue
                
                # if args.config.normalization_type == 'log': 
                #     log_pred   = mp.predicted
                #     log_target = mp.ground_truth
                # else:
                #     log_pred   = symlog(mp.predicted)
                #     log_target = symlog(mp.ground_truth)
                log_pred   = symlog(mp.predicted)
                log_target = symlog(mp.ground_truth)

                self.log_maes[key](log_pred, log_target)
                metrics[f"{key}_log_mae"] = self.log_maes[key]

                if self.config.report_rmse:
                    self.log_rmses[key](log_pred, log_target)
                    metrics[f"{key}_log_rmse"] = self.log_rmses[key]

                if self.config.report_mape:
                    self.log_mapes[key](log_pred, log_target)
                    metrics[f"{key}_log_mape"] = self.log_mapes[key]

                if self.config.report_smape:
                    self.log_smapes[key](log_pred, log_target)
                    metrics[f"{key}_log_smape"] = self.log_smapes[key]

                if self.config.report_mse:
                    self.log_mses[key](log_pred, log_target)
                    metrics[f"{key}_log_mse"] = self.log_mses[key]


        for key, cls_metric in self.cls_metrics.items():
            if (mp := self.cls_provider(key, batch, preds)) is None:
                continue

            metric_dict = cls_metric(mp.predicted, mp.ground_truth)
            metrics.update(
                {
                    f"{key}_{metric_name}": metric
                    for metric_name, metric in metric_dict.items()
                }
            )

        if self.config.check_conflicting_structures is not None:
            for name, conflicting in self.conflicting.items():
                metric_dict = conflicting(batch, preds)
                metrics.update(
                    {
                        f"conflicting/{name}_{metric_name}": metric
                        for metric_name, metric in metric_dict.items()
                    }
                )

        return metrics

class MTFinetuneMetrics(nn.Module):
    
    def regression_targets(self, task: "TaskConfig"):
        return (
            task.graph_scalar_targets + task.node_vector_targets
            + [target.name for target in task.graph_classification_adabin_targets]
        )

    @override
    def __init__(
        self,
        config: MetricsConfig,
        provider: MTMetricPairProvider,
        cls_provider: MTMetricPairProvider | None,
        tasks: list["TaskConfig"],
        roi_getter: Callable[[str], tuple[float, float]] | None = None,
    ):
        super().__init__()

        if not isinstance(provider, MTMetricPairProvider):
            raise TypeError(
                f"Expected {provider=} to be an instance of {MTMetricPairProvider=}"
            )
        self.provider = provider
        self.cls_provider = cls_provider
        self.roi_getter = roi_getter

        
        self.config = config
        self.tasks = tasks

        self._metrics = nn.ModuleDict()
        for t_idx, task in enumerate(tasks):
            prefix = f"{task.name}/"

            for target in self.regression_targets(task):
                self._metrics[prefix + target + "_mae"] = torchmetrics.MeanAbsoluteError()
                if config.report_normal_metrics:
                    if config.report_rmse:
                        self._metrics[prefix + target + "_rmse"] = torchmetrics.MeanSquaredError(squared=False)
                    if config.report_mape:
                        self._metrics[prefix + target + "_mape"] = torchmetrics.MeanAbsolutePercentageError()
                    if config.report_smape:
                        self._metrics[prefix + target + "_smape"] = torchmetrics.SymmetricMeanAbsolutePercentageError()
                    if config.report_roi_smape:
                        self._metrics[prefix + target + "_roi_smape"] = torchmetrics.SymmetricMeanAbsolutePercentageError()
                    if config.report_mse:
                        self._metrics[prefix + target + "_mse"] = torchmetrics.MeanSquaredError(squared=True)

                if config.report_log_metrics:
                    self._metrics[prefix + target + "_log_mae"] = torchmetrics.MeanAbsoluteError()
                    if config.report_rmse:
                        self._metrics[prefix + target + "_log_rmse"] = torchmetrics.MeanSquaredError(squared=False)
                    if config.report_mape:
                        self._metrics[prefix + target + "_log_mape"] = torchmetrics.MeanAbsolutePercentageError()
                    if config.report_smape:
                        self._metrics[prefix + target + "_log_smape"] = torchmetrics.SymmetricMeanAbsolutePercentageError()
                    if config.report_mse:
                        self._metrics[prefix + target + "_log_mse"] = torchmetrics.MeanSquaredError(squared=True)
                

            # TODO: Support classification targets
            

    def forward(self, batch: BaseData, preds: dict[str, torch.Tensor]):
        metrics: dict[str, torchmetrics.Metric] = {}

        for t_idx, task in enumerate(self.tasks):
            prefix = f"{task.name}/"

            if self.config.report_normal_metrics:
                for key in self.regression_targets(task):
                    mp = self.provider(key, batch, preds, t_idx)
                    if mp is None:
                        continue
                    for suf in ["_mae", "_rmse", "_mape", "_smape", "_roi_smape", "_mse"]:
                        mkey = prefix + key + suf
                        if mkey in self._metrics.keys():
                            if suf == "_roi_smape":
                                if self.roi_getter is None:
                                    raise RuntimeError(
                                        f"ROI-SMAPE requested (metric '{mkey}') but no roi_getter was provided "
                                        f"to MTFinetuneMetrics. Pass roi_getter=model._get_roi when constructing metrics."
                                    )
    
                                # --- ROI SMAPE ---
                                roi_lo, roi_hi = self.roi_getter(key)
                                mask = (mp.ground_truth >= roi_lo) & (mp.ground_truth <= roi_hi)

                                if mask.any():
                                    self._metrics[mkey](mp.predicted[mask], mp.ground_truth[mask])
                            else:
                                self._metrics[mkey](mp.predicted, mp.ground_truth)
                            
                            metrics[mkey] = self._metrics[mkey]

            if self.config.report_log_metrics:
                for key in self.regression_targets(task):
                    mp = self.provider(key, batch, preds, t_idx)
                    if mp is None:
                        continue
                    log_pred   = symlog(mp.predicted)
                    log_target = symlog(mp.ground_truth)
                    for suf in ["_log_mae", "_log_rmse", "_log_mape", "_log_smape", "_log_mse"]:
                        mkey = prefix + key + suf
                        if mkey in self._metrics:
                            self._metrics[mkey](log_pred, log_target)
                            metrics[mkey] = self._metrics[mkey]

        return metrics