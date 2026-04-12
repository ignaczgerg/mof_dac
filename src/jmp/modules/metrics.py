"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections.abc import Callable
from functools import partial
from typing import TypedDict

import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.classification
from jmp.lightning.util.typed import TypedModuleList
from torch_geometric.data import Batch
from typing_extensions import NotRequired, override

from .transforms.normalize import denormalize_batch
from .transforms.units import VALID_UNITS, Unit, _determine_factor


class MetricConfig(TypedDict):
    idx: int
    additional_units: NotRequired[list[str]]
    num_ctx_atoms: int | float 


def _transform(x: torch.Tensor, *, from_: Unit, to: Unit):
    factor = _determine_factor(from_, to)
    return x * factor


class FMTaskMetrics(nn.Module):
    @override
    def __init__(
        self,
        name: str,
        config: MetricConfig,
        num_tasks: int,
        free_atoms_only: bool = True,
        num_classes: int | None = None,
    ):
        super().__init__()

        self.name = name
        self.config = config
        self.num_tasks = num_tasks
        self.free_atoms_only = free_atoms_only

        self.energy_mae = torchmetrics.MeanAbsoluteError()
        self.forces_mae = torchmetrics.MeanAbsoluteError()
        self.positions_mae = torchmetrics.MeanAbsoluteError()
        if num_classes is not None:
            self.atomic_numbers_accuracy = torchmetrics.classification.MulticlassAccuracy(
                average="macro", num_classes=num_classes
            )

        if units := self.config.get("additional_units", []):
            for unit in units:
                if unit not in VALID_UNITS:
                    raise ValueError(
                        f"Invalid unit: {unit}. Valid units: {VALID_UNITS}"
                    )
            self.energy_mae_additional = TypedModuleList(
                [torchmetrics.MeanAbsoluteError() for _ in units]
            )
            self.forces_mae_additional = TypedModuleList(
                [torchmetrics.MeanAbsoluteError() for _ in units]
            )

    @override
    def forward(self, batch: Batch, energy: torch.Tensor, forces: torch.Tensor | None,
                positions: torch.Tensor | None = None, 
                atomic_numbers: torch.Tensor | None = None):
        metrics: dict[str, torchmetrics.Metric] = {}

        if energy is not None:
            self._energy_mae(batch, energy, self.energy_mae)
            metrics["energy_mae"] = self.energy_mae
        if forces is not None:
            self._forces_mae(batch, forces, self.forces_mae)
            metrics["forces_mae"] = self.forces_mae
        if positions is not None:
            self._positions_mae(batch, positions, self.positions_mae)
            metrics["positions_mae"] = self.positions_mae
        if atomic_numbers is not None:
            self._atomic_numbers_accuracy(
                batch, atomic_numbers, self.atomic_numbers_accuracy
            )
            metrics["atomic_numbers_accuracy"] = self.atomic_numbers_accuracy
            


        if additional := self.config.get("additional_units", []):
            for unit, energy_metric, forces_metric in zip(
                additional, self.energy_mae_additional, self.forces_mae_additional
            ):
                assert (
                    unit in VALID_UNITS
                ), f"Invalid unit: {unit}. Valid units: {VALID_UNITS}"
                sanitized_unit = unit.replace("/", "_")
                self._energy_mae(
                    batch,
                    energy,
                    energy_metric,
                    transform=partial(_transform, from_="eV", to=unit),
                )
                self._forces_mae(
                    batch,
                    forces,
                    forces_metric,
                    transform=partial(_transform, from_="eV", to=unit),
                )

                metrics[f"energy_mae_{sanitized_unit}"] = energy_metric
                metrics[f"forces_mae_{sanitized_unit}"] = forces_metric

        return {f"{self.name}/{name}": metric for name, metric in metrics.items()}

    def _forces_mae(
        self,
        batch: Batch,
        forces: torch.Tensor,
        forces_mae: torchmetrics.MeanAbsoluteError,
        *,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        task_idx = self.config["idx"]

        forces_mask = batch.task_mask[:, task_idx]  # (b,)
        forces_mask = forces_mask[batch.batch]  # (n,)
        if self.free_atoms_only:
            forces_mask = forces_mask & ~batch.fixed
        forces_target = batch.force[..., task_idx][forces_mask]
        forces_pred = forces[..., task_idx][forces_mask]
        if transform is not None:
            forces_target = transform(forces_target)
            forces_pred = transform(forces_pred)

        forces_mae(forces_pred, forces_target)

    def _energy_mae(
        self,
        batch: Batch,
        energy: torch.Tensor,
        energy_mae: torchmetrics.MeanAbsoluteError,
        *,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        task_idx = self.config["idx"]

        energy_mask = batch.task_mask[:, task_idx]  # (b,)
        energy_target = batch.y[..., task_idx][energy_mask]  # (b,)
        energy_pred = energy[..., task_idx][energy_mask]  # (b,)
        if transform is not None:
            energy_target = transform(energy_target)
            energy_pred = transform(energy_pred)

        energy_mae(energy_pred, energy_target)
        
        
        
    def _positions_mae(
        self,
        batch: Batch,
        positions: torch.Tensor,
        positions_mae: torchmetrics.MeanAbsoluteError,
    ):

        task_idx = self.config["idx"]

        pred = positions[..., task_idx]       # (n, 3)
        target = batch.pos[..., task_idx]     # (n, 3)

        # build next-atom mask & shifted targets
        N = pred.size(0)
        device = pred.device
        valid = torch.zeros(N, dtype=torch.bool, device=device)
        shifted_target = torch.zeros_like(pred)

        num_ctx = self.config["num_ctx_atoms"]
        ptr = batch.ptr

        for gi in range(batch.num_graphs):
            start, end = int(ptr[gi]), int(ptr[gi+1])
            skip = max(1, int(num_ctx if num_ctx >= 1 else num_ctx * (end-start)))

            # valid j run from j_start…j_end-1
            # j_start = start if skip == 0 else (start + skip - 1)
            j_start = start + skip - 1
            j_end = end - 1

            if j_end > j_start:
                valid[j_start:j_end] = True
                shifted_target[j_start:j_end] = target[j_start+1 : j_end+1]

        # per-task atom mask, check forces MAE
        task_mask = batch.task_mask[:, task_idx]    # (b,)
        atom_task_mask = task_mask[batch.batch]     # (n,)
        
        if getattr(self, "free_atoms_only", False):
            atom_task_mask &= ~batch.fixed

        mask = valid & atom_task_mask         # (m,)

        pred_sel   = pred[mask]               # (m, 3)
        target_sel = shifted_target[mask]     # (m, 3)
        positions_mae(pred_sel, target_sel)
        

    def _atomic_numbers_accuracy(
        self,
        batch: Batch,
        atomic_numbers_pred: torch.Tensor,  # (n, num_classes, t)
        atomic_numbers_accuracy: torchmetrics.classification.MulticlassAccuracy,
    ):

        task_idx = self.config["idx"]

        logits = atomic_numbers_pred[..., task_idx]     # (n, num_classes)
        pred_class = logits.argmax(dim=1)               # (n,)
        target = batch.atomic_numbers[..., task_idx]    # (n,)

        N = pred_class.size(0)
        device = pred_class.device
        valid = torch.zeros(N, dtype=torch.bool, device=device)
        shifted_target = torch.zeros(N, dtype=torch.long, device=device)

        num_ctx = self.config["num_ctx_atoms"]
        ptr = batch.ptr  # (b+1,)

        for gi in range(batch.num_graphs):
            start, end = int(ptr[gi]), int(ptr[gi+1])
            skip = max(1, int(num_ctx) if num_ctx >= 1 else int(num_ctx * (end - start)))

            # valid j run from j_start…j_end-1
            # j_start = start if skip == 0 else (start + skip - 1)
            j_start = start + skip - 1
            j_end   = end - 1

            if j_end > j_start:
                valid[j_start:j_end] = True
                shifted_target[j_start:j_end] = target[j_start+1 : j_end+1]

        task_mask = batch.task_mask[:, task_idx]   # (b,)
        atom_task_mask = task_mask[batch.batch]    # (n,)
        valid &= atom_task_mask
        if getattr(self.config, "train_on_free_atoms_only", False):
            valid &= ~batch.fixed               

        preds   = pred_class[valid]              
        targets = shifted_target[valid]          
        
        # only update if there’s at least one example
        if preds.numel() > 0:
            atomic_numbers_accuracy(preds, targets)
        
    # def _atomic_numbers_accuracy(
    #     self,
    #     batch: Batch,
    #     atomic_numbers: torch.Tensor,
    #     atomic_numbers_accuracy: torchmetrics.classification.MulticlassAccuracy,
    # ):
    #     task_idx = self.config["idx"]
        
    #     atomic_numbers_mask = batch.task_mask[:, task_idx]
    #     atomic_numbers_mask = atomic_numbers_mask[batch.batch]
    #     if self.free_atoms_only:
    #         atomic_numbers_mask = atomic_numbers_mask & ~batch.fixed
    #     atomic_numbers_target = batch.atomic_numbers[..., task_idx][atomic_numbers_mask]
    #     atomic_numbers_pred = atomic_numbers[..., task_idx][atomic_numbers_mask]
    #     atomic_numbers_pred = atomic_numbers_pred.argmax(dim=-1)  # (n,)

    #     atomic_numbers_accuracy(atomic_numbers_pred, atomic_numbers_target)
        
        
        
class FMMetrics(nn.Module):
    @override
    def __init__(
        self,
        tasks: dict[str, MetricConfig],
        *,
        denormalize: bool,
        free_atoms_only: bool = True,
        num_classes: int | None = None,
    ):
        super().__init__()

        self.denormalize = denormalize
        self.task_metrics = TypedModuleList(
            [
                FMTaskMetrics(
                    name, config, num_tasks=len(tasks), free_atoms_only=free_atoms_only, 
                    num_classes=num_classes
                )
                for name, config in tasks.items()
            ]
        )

    @override
    def forward(self, batch: Batch, energy: torch.Tensor, forces: torch.Tensor | None, 
                positions: torch.Tensor | None = None, 
                atomic_numbers: torch.Tensor | None = None):
        
        tensors_to_denorm = {}
        if self.denormalize:
            if energy is not None:
                tensors_to_denorm["y"] = energy
            if positions is not None:
                tensors_to_denorm["pos"] = positions
            if forces is not None:
                tensors_to_denorm["force"] = forces

            batch, d = denormalize_batch(batch, tensors_to_denorm)
            energy = d.get("y", energy)
            forces = d.get("force", forces)
            positions = d.get("pos", positions)

        metrics: dict[str, torchmetrics.Metric] = {}
        for task_metrics in self.task_metrics:
            metrics.update(task_metrics(batch, energy, forces, positions, atomic_numbers))
        return metrics
