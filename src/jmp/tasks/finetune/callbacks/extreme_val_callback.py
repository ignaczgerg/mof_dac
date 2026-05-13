"""Callback that evaluates the model on a small set of 'extreme' MOF CIFs each
validation epoch and logs separate metrics (MAE, RMSE, R²) to the trainer's
logger under the ``extreme/`` prefix.

Usage:
    callback = ExtremeValCallback(
        cif_dirs=["temp/Firas/batch_5", "temp/Firas/batch_6"],
        csv_path="temp/Firas/extreme_mofs_targets.csv",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader

from jmp.modules.transforms.normalize import denormalize_tensor

log = logging.getLogger(__name__)

# Lazily imported so the module can be imported without the inference utils on
# the Python path (they live under my_scripts/).
_CIFFolderDataset = None


def _get_cif_folder_dataset():
    global _CIFFolderDataset
    if _CIFFolderDataset is None:
        import sys

        # Ensure my_scripts/inference is importable
        scripts_dir = str(
            Path(__file__).resolve().parents[5] / "my_scripts" / "inference"
        )
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from utils import CIFFolderDataset

        _CIFFolderDataset = CIFFolderDataset
    return _CIFFolderDataset


class ExtremeValCallback(Callback):
    """Run inference on extreme-uptake CIFs and log metrics each val epoch."""

    def __init__(
        self,
        cif_dirs: Sequence[str | Path],
        csv_path: str | Path,
        target: str = "co2_uptake",
        id_col: str | None = None,
        batch_size: int = 4,
        num_workers: int = 0,
    ):
        super().__init__()
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers

        # --- Discover CIF paths ---
        all_paths: list[Path] = []
        for d in cif_dirs:
            d = Path(d)
            if d.is_dir():
                all_paths.extend(sorted(d.glob("*.cif")))
            else:
                log.warning("ExtremeValCallback: directory %s not found, skipping", d)
        if not all_paths:
            raise FileNotFoundError(
                f"No CIF files found in directories: {list(cif_dirs)}"
            )
        self._cif_paths = all_paths
        log.info("ExtremeValCallback: found %d CIF files", len(all_paths))

        # --- Load ground-truth CSV ---
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        # Auto-detect ID column
        if id_col is None:
            for candidate in ("id", "name", "cif", "sid"):
                if candidate in df.columns:
                    id_col = candidate
                    break
            if id_col is None:
                id_col = df.columns[0]
        if target not in df.columns:
            raise KeyError(
                f"Target column '{target}' not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
        self._gt: dict[str, float] = dict(
            zip(df[id_col].astype(str), df[target].astype(float))
        )
        log.info(
            "ExtremeValCallback: loaded %d ground-truth entries from %s",
            len(self._gt),
            csv_path,
        )

        self._dataset = None  # built lazily

    # --------------------------------------------------------------------- #
    # Lightning hook
    # --------------------------------------------------------------------- #
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.sanity_checking:
            return
        if trainer.global_rank != 0:
            return
        # Only meaningful when the model predicts the target
        if self.target not in getattr(pl_module.config, "graph_scalar_targets", []):
            return

        dataset = self._get_or_build_dataset(pl_module)
        if dataset is None or len(dataset) == 0:
            return

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=pl_module.collate_fn,
        )

        device = pl_module.device
        preds_list: list[torch.Tensor] = []
        gt_list: list[torch.Tensor] = []
        sids: list[str] = []

        pl_module.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = pl_module(batch)  # dict[str, Tensor(B, T)]
                raw = out.get(self.target)
                if raw is None:
                    continue
                # All task columns are identical (shared head); take column 0
                pred = raw[:, 0]  # (B,)

                # Denormalize
                pred = self._denormalize(pred, pl_module)

                # Inverse uptake conversion to mmol/g
                pred = self._inverse_uptake(pred, batch, pl_module)

                # Gather ground-truth for matched sids
                batch_sids = batch.sid if isinstance(batch.sid, list) else [batch.sid]
                for i, sid in enumerate(batch_sids):
                    if sid in self._gt:
                        preds_list.append(pred[i : i + 1])
                        gt_list.append(
                            torch.tensor([self._gt[sid]], device=device)
                        )
                        sids.append(sid)

        if not preds_list:
            log.warning("ExtremeValCallback: no matched predictions")
            return

        all_pred = torch.cat(preds_list)
        all_gt = torch.cat(gt_list)

        mae = (all_pred - all_gt).abs().mean().item()
        rmse = ((all_pred - all_gt) ** 2).mean().sqrt().item()
        ss_res = ((all_gt - all_pred) ** 2).sum()
        ss_tot = ((all_gt - all_gt.mean()) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")
        denom = (all_pred.abs() + all_gt.abs()).clamp(min=1e-8)
        smape = (2.0 * (all_pred - all_gt).abs() / denom).mean().item() * 100.0

        # Log metrics
        pl_module.log(f"extreme/{self.target}_mae", mae, prog_bar=False, rank_zero_only=True)
        pl_module.log(f"extreme/{self.target}_rmse", rmse, prog_bar=False, rank_zero_only=True)
        pl_module.log(f"extreme/{self.target}_r2", r2, prog_bar=False, rank_zero_only=True)
        pl_module.log(f"extreme/{self.target}_smape", smape, prog_bar=False, rank_zero_only=True)

        log.info(
            "ExtremeValCallback epoch %d: MAE=%.4f  RMSE=%.4f  R2=%.4f  SMAPE=%.2f%%  (n=%d)",
            trainer.current_epoch, mae, rmse, r2, smape, len(all_pred),
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _get_or_build_dataset(self, pl_module):
        if self._dataset is not None:
            return self._dataset
        CIFFolderDataset = _get_cif_folder_dataset()
        # Use the first CIF's parent as cif_dir (required param)
        cif_dir = self._cif_paths[0].parent
        geo_props = None
        fusion_names = None
        fmf = getattr(pl_module.config, "mof_feature_fusion", None)
        if fmf and fmf.get("enabled", False):
            fusion_names = fmf.get("feature_names", [])
            try:
                    from utils import compute_geometric_properties_batch
                    geo_props = compute_geometric_properties_batch(
                        [str(p) for p in self._cif_paths]
                    )
            except Exception as e:
                    log.warning("ExtremeValCallback: could not compute geo props: %s", e)
        raw_dataset = CIFFolderDataset(
            cif_dir=cif_dir,
            paths=self._cif_paths,
            sid_from="stem",
            geometric_properties=geo_props,
            fusion_feature_names=fusion_names,
        )
        # Apply model transforms (graph construction, edge indices, etc.)
        self._dataset = pl_module._apply_dataset_transforms(raw_dataset)
        return self._dataset

    def _denormalize(self, pred: torch.Tensor, pl_module) -> torch.Tensor:
        """Denormalize predictions using the first task's normalization config."""
        for task in pl_module.config.tasks:
            if task.normalization and self.target in task.normalization:
                norm = task.normalization[self.target]
                return denormalize_tensor(pred, norm)
        return pred

    def _inverse_uptake(
        self, pred: torch.Tensor, batch, pl_module
    ) -> torch.Tensor:
        """Convert predictions back from per_cell / per_volume to mmol/g."""
        from ase.data import atomic_masses

        mode = getattr(pl_module.config, "uptake_conversion_mode", "none")
        if mode == "none":
            return pred

        # Compute per-structure molecular weights
        # batch.atomic_numbers is concatenated; use batch.batch to split
        batch_idx = batch.batch  # (total_atoms,)
        num_structures = int(batch_idx.max().item()) + 1
        mw = torch.zeros(num_structures, device=pred.device)
        for s in range(num_structures):
            mask = batch_idx == s
            zs = batch.atomic_numbers[mask].cpu().numpy()
            mw[s] = float(atomic_masses[zs].sum())

        if mode == "per_cell":
            pred = pred * 1000.0 / mw
        elif mode == "per_volume":
            # Compute cell volume per structure
            # batch.cell is (num_structures, 3, 3)
            cell = batch.cell
            if cell.dim() == 3:
                vol = torch.det(cell).abs()
            else:
                vol = torch.det(cell.view(-1, 3, 3)).abs()
            pred = pred * 1000.0 * vol / mw
        return pred
