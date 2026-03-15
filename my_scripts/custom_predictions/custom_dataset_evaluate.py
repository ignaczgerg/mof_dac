import argparse
from pathlib import Path
import copy
from typing import Mapping, Any, List

import numpy as np
import torch
from argparse import Namespace

from jmp.lightning import Trainer
from jmp.tasks.finetune.adsorption_db import AdsorptionDbModel, AdsorptionDbConfig, AdsorptionFilter
from jmp.tasks.finetune.base import FinetuneLmdbDatasetConfig
from jmp.tasks.finetune.adsorption_db import TaskConfig
from jmp.modules.dataset.common import wrap_common_dataset
from jmp.modules.dataset import dataset_transform as DT
from tqdm import tqdm
from torch_geometric.data import Data as PyGData, Batch as PyGBatch


def _safe_collate_fn(model):
    base = model.collate_fn

    def _wrap(batch):
        if isinstance(batch, (PyGData, PyGBatch)):
            batch = [batch]
        if not isinstance(batch, list):
            try:
                batch = list(batch)
            except TypeError:
                batch = [batch]
        return base(batch)

    return _wrap


def _strip_args(ns: Namespace, keys=("name", "regress_energy")) -> Namespace:
    ns = copy.deepcopy(ns)
    for k in keys:
        if hasattr(ns, k):
            delattr(ns, k)
    return ns


def _clone_lmdb_cfg_from_template(template_ds_cfg, src_path: Path, args_ns: Namespace):
    args_ns = _strip_args(args_ns)
    if isinstance(template_ds_cfg, FinetuneLmdbDatasetConfig):
        new_cfg = copy.deepcopy(template_ds_cfg)
        new_cfg.src = Path(src_path)
        new_cfg.args = args_ns
        return new_cfg
    return FinetuneLmdbDatasetConfig(src=Path(src_path), args=args_ns)


def _namespacefy_args(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        obj = {k: _namespacefy_args(v) for k, v in obj.items()}
        if "args" in obj and isinstance(obj["args"], Mapping):
            obj["args"] = Namespace(**obj["args"])
        return obj
    if isinstance(obj, list):
        return [_namespacefy_args(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_namespacefy_args(v) for v in obj)
    return obj


def _extract_config_dict(ckpt: dict) -> dict:
    hp = ckpt.get("hyper_parameters", {})
    if isinstance(hp, Mapping) and "hparams" in hp and isinstance(hp["hparams"], Mapping):
        cfg_dict = hp["hparams"]
    elif isinstance(hp, Mapping) and "config" in hp and isinstance(hp["config"], Mapping):
        cfg_dict = hp["config"]
    elif isinstance(hp, Mapping):
        cfg_dict = hp
    else:
        raise RuntimeError("No usable hyper-parameters found in checkpoint.")
    return dict(cfg_dict)


def _get_attr(d, *names, default=None):
    if isinstance(d, dict):
        for n in names:
            if n in d and d[n] is not None:
                return d[n]
    else:
        for n in names:
            v = getattr(d, n, None)
            if v is not None:
                return v
    return default


def _denorm(value: torch.Tensor, norm_cfg: Any | None) -> torch.Tensor:
    if norm_cfg is None:
        return value

    mean = _get_attr(norm_cfg, "mean", default=None)
    std = _get_attr(norm_cfg, "std", default=None)
    ntype = _get_attr(norm_cfg, "normalization_type", "type", default="standard")

    if std is not None:
        value = value * torch.as_tensor(std, device=value.device, dtype=value.dtype)
    if mean is not None:
        value = value + torch.as_tensor(mean, device=value.device, dtype=value.dtype)

    ntype_str = str(ntype).lower()
    if "log" in ntype_str:
        finfo = torch.finfo(value.dtype)
        max_log = torch.log(torch.tensor(finfo.max, device=value.device)) - 1.0
        value = torch.clamp(value, max=max_log)
        value = torch.exp(value)
    return value


def _sid_to_list(sid_attr) -> list:
    if isinstance(sid_attr, torch.Tensor):
        return sid_attr.detach().cpu().view(-1).tolist()
    if isinstance(sid_attr, (list, tuple)):
        out = []
        for x in sid_attr:
            if isinstance(x, torch.Tensor):
                out.extend(x.detach().cpu().view(-1).tolist())
            elif isinstance(x, (list, tuple)):
                out.extend(_sid_to_list(x))
            else:
                out.append(x)
        return out
    return [sid_attr]


def _head_names_and_norms(cfg: AdsorptionDbConfig) -> tuple[list[str], list[dict | None]]:
    train_tasks = getattr(cfg, "train_tasks", None)
    if train_tasks:
        names = [t.name for t in train_tasks]
        norms = [getattr(t, "normalization", None) for t in train_tasks]
        return names, norms

    tasks = getattr(cfg, "tasks", None)
    if tasks:
        names = [t.name for t in tasks]
        norms = [getattr(t, "normalization", None) for t in tasks]
        return names, norms

    names = [getattr(cfg, "name", "train_task")]
    norms = [getattr(cfg, "normalization", None)]
    return names, norms


def _resolve_forced_norm(
    head_names: list[str],
    head_norms: list[dict | None],
    selector: str | None,
) -> list[dict | None]:
    if not selector:
        return head_norms

    key = selector.strip().lower()
    aliases = {
        "db1": ["db1", "adsorption_db_1", "adsorption_db1", "db_1"],
        "db2": ["db2", "adsorption_db_2", "adsorption_db2", "db_2"],
        # "cof2": ["cof3", "adsorption_cof_db_2", "adsorption_cof2", "cof_2"],
    }
    candidates = aliases.get(key, [key])

    forced = None
    for name, norm in zip(head_names, head_norms):
        lname = name.lower()
        if any(c == lname or c in lname for c in candidates):
            forced = norm
            break

    if forced is None:
        return head_norms

    return [forced for _ in head_norms]

def _predict_split(
    split: str,
    model: AdsorptionDbModel,
    device: torch.device,
    out_dir: Path,
    head_names: List[str],
    head_norms: List[dict | None],
    ext_ds_cfg: FinetuneLmdbDatasetConfig | None = None,
    suffix: str = "",
    num_workers: int = 6,
    max_natoms: int | None = None,
):
    assert split in ("val", "test")
    model.eval().to(device)

    if ext_ds_cfg is not None:
        ds = ext_ds_cfg.create_dataset(split)
        ds = wrap_common_dataset(ds, ext_ds_cfg)
        # Apply max_natoms filtering for external datasets
        if max_natoms is not None:
            ds = DT.filter_transform(ds, predicate=AdsorptionFilter(max_natoms=max_natoms))
        ds = model._apply_dataset_transforms(ds)
    else:
        ds = model.val_dataset() if split == "val" else model.test_dataset()
    if ds is None:
        return

    bs = getattr(
        model.config,
        "test_batch_size",
        getattr(model.config, "eval_batch_size", getattr(model.config, "batch_size", 4)),
    )

    num_workers = int(num_workers)
    kwargs = dict(
        dataset=ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_safe_collate_fn(model),
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 1

    dl = torch.utils.data.DataLoader(**kwargs)

    acc = {
        head_name: {t: {"sid": [], "pred": []} for t in model.config.graph_scalar_targets}
        for head_name in head_names
    }

    with torch.no_grad():
        for batch in tqdm(dl, desc=f"Predicting {split}", unit="batch"):
            batch = batch.to(device, non_blocking=True)
            preds = model(batch)
            B = batch.num_graphs

            if hasattr(batch, "sid"):
                sids = _sid_to_list(getattr(batch, "sid"))
                if len(sids) != B:
                    sids = sids[:B]
            else:
                sids = list(range(B))

            for target in model.config.graph_scalar_targets:
                if target not in preds:
                    continue

                y = preds[target]
                y = y.view(B, -1) if (y.ndim >= 2 and y.shape[-1] > 1) else y.view(B, 1)
                C = y.shape[-1]

                for c in range(min(C, len(head_names))):
                    norm_dict = head_norms[c] or {}
                    norm_cfg = norm_dict.get(target) if isinstance(norm_dict, dict) else None
                    y_c = _denorm(y[:, c], norm_cfg)

                    head_name = head_names[c]
                    acc[head_name][target]["sid"].extend(sids)
                    acc[head_name][target]["pred"].extend(y_c.detach().cpu().tolist())

    if suffix:
        out_dir = out_dir.parent / f"{out_dir.name}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for head_name, task_acc in acc.items():
        short_name = head_name.replace("adsorption_", "")
        for target, d in task_acc.items():
            if not d["sid"]:
                continue
            sid = np.array(d["sid"], dtype=object)
            pred = np.array(d["pred"], dtype=np.float32)
            np.savez_compressed(out_dir / f"{target}_{short_name}.npz", sid=sid, pred=pred)


def main():
    parser = argparse.ArgumentParser("Export predictions (val+test) from a fine-tuned checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to fine-tuned .ckpt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--val_src", type=Path, default=None, help="LMDB dir for external VAL")
    parser.add_argument("--test_src", type=Path, default=None, help="LMDB dir for external TEST")
    parser.add_argument("--template_task", type=int, default=0, help="Index of template task to copy settings from")
    parser.add_argument("--task_name", type=str, default="external", help="Name of the injected task/dataset")
    parser.add_argument("--only_new_task", action="store_true", help="Evaluate only the injected dataset")
    parser.add_argument(
        "--predict_only",
        action="store_true",
        help="Run pure inference on external LMDB(s) without metrics/loss.",
    )
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for the output folder")
    parser.add_argument(
        "--denorm_task",
        type=str,
        default=None,
        help="Force denorm using μ/σ of a specific training task (e.g., 'db1', 'db2').",
    )
    parser.add_argument(
        "--max_natoms",
        type=int,
        default=None,
        help="Filter structures with more than N atoms",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    ckpt_dir = ckpt_path.parent

    raw = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = _extract_config_dict(raw)
    cfg_dict = _namespacefy_args(cfg_dict)
    cfg = AdsorptionDbConfig(**cfg_dict)
    model = AdsorptionDbModel(cfg)
    model.load_state_dict(raw["state_dict"], strict=True)

    val_ds_cfg = None
    test_ds_cfg = None
    template = None
    if args.val_src is not None or args.test_src is not None:
        cfg = model.config
        if getattr(cfg, "tasks", None):
            t = max(0, min(args.template_task, len(cfg.tasks) - 1))
            template = cfg.tasks[t]
            template_val_ds = template.val_dataset
            template_test_ds = template.test_dataset
        else:
            template = None
            template_val_ds = cfg.val_dataset
            template_test_ds = cfg.test_dataset

        if args.val_src is not None:
            base_args = getattr(template_val_ds, "args", cfg.args)
            val_ds_cfg = _clone_lmdb_cfg_from_template(template_val_ds, args.val_src, base_args)
        if args.test_src is not None:
            base_args = getattr(template_test_ds, "args", cfg.args)
            test_ds_cfg = _clone_lmdb_cfg_from_template(template_test_ds, args.test_src, base_args)

        if not args.predict_only:
            def _mk_ds(path_like, fallback_args):
                return FinetuneLmdbDatasetConfig(src=Path(path_like), args=_strip_args(fallback_args))

            val_src = args.val_src or args.test_src
            test_src = args.test_src
            val_args = getattr(template_val_ds, "args", cfg.args) if template_val_ds else cfg.args
            test_args = getattr(template_test_ds, "args", cfg.args) if template_test_ds else cfg.args

            template_for_fields = template if template is not None else TaskConfig(
                name=getattr(cfg, "name", "train_task"),
                train_dataset=None,
                val_dataset=cfg.val_dataset,
                test_dataset=cfg.test_dataset,
                graph_scalar_targets=list(cfg.graph_scalar_targets),
                graph_classification_targets=list(cfg.graph_classification_targets),
                graph_classification_adabin_targets=list(cfg.graph_classification_adabin_targets),
                node_vector_targets=list(cfg.node_vector_targets),
                graph_scalar_loss_coefficients=dict(cfg.graph_scalar_loss_coefficients),
                graph_classification_loss_coefficients=dict(cfg.graph_classification_loss_coefficients),
                node_vector_loss_coefficients=dict(cfg.node_vector_loss_coefficients),
                normalization=dict(cfg.normalization) if cfg.normalization else None,
            )

            new_task = TaskConfig(
                name=args.task_name,
                train_dataset=None,
                val_dataset=_mk_ds(val_src, val_args) if val_src else None,
                test_dataset=_mk_ds(test_src, test_args) if test_src else None,
                graph_scalar_targets=list(template_for_fields.graph_scalar_targets),
                graph_classification_targets=list(template_for_fields.graph_classification_targets),
                graph_classification_adabin_targets=list(template_for_fields.graph_classification_adabin_targets),
                node_vector_targets=list(getattr(template_for_fields, "node_vector_targets", [])),
                graph_scalar_loss_coefficients=dict(template_for_fields.graph_scalar_loss_coefficients),
                graph_classification_loss_coefficients=dict(
                    template_for_fields.graph_classification_loss_coefficients
                ),
                node_vector_loss_coefficients=dict(template_for_fields.node_vector_loss_coefficients),
                normalization=(dict(template_for_fields.normalization) if template_for_fields.normalization else None),
            )

            if getattr(cfg, "tasks", None):
                if args.only_new_task:
                    cfg.tasks = [new_task]
                    cfg.train_tasks = []
                else:
                    cfg.tasks = [*cfg.tasks, new_task]
            else:
                cfg.val_dataset = new_task.val_dataset
                cfg.test_dataset = new_task.test_dataset

    head_names, head_norms_base = _head_names_and_norms(model.config)
    fallback = getattr(model.config, "normalization", None) or {}
    head_norms = [norm or fallback for norm in head_norms_base]
    head_norms = _resolve_forced_norm(head_names, head_norms, args.denorm_task)

    model.config.args.log_predictions = True
    if args.max_natoms is not None:
        model.config.args.max_natoms = args.max_natoms
    try:
        model.config.trainer.num_sanity_val_steps = 0
        model.config.trainer.logging.wandb.enabled = False
    except Exception:
        pass

    if args.num_workers is not None:
        model.config.num_workers = int(args.num_workers)
        try:
            model.config.args.num_workers = int(args.num_workers)
        except Exception:
            pass

    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )


    if args.predict_only:
        base = Path(ckpt_dir) / "predictions_external" / args.task_name
        ran = False
        if args.val_src is not None:
            _predict_split(
                "val",
                model,
                device,
                base / "val",
                head_names,
                head_norms,
                ext_ds_cfg=val_ds_cfg,
                suffix=args.suffix,
                num_workers=args.num_workers,
                max_natoms=args.max_natoms,
            )
            ran = True
        if args.test_src is not None:
            _predict_split(
                "test",
                model,
                device,
                base / "test",
                head_names,
                head_norms,
                ext_ds_cfg=test_ds_cfg,
                suffix=args.suffix,
                num_workers=args.num_workers,
                max_natoms=args.max_natoms,
            )
            ran = True
        if not ran:
            _predict_split(
                "val",
                model,
                device,
                base / "val",
                head_names,
                head_norms,
                ext_ds_cfg=None,
                suffix=args.suffix,
                num_workers=args.num_workers,
                max_natoms=args.max_natoms,
            )
            _predict_split(
                "test",
                model,
                device,
                base / "test",
                head_names,
                head_norms,
                ext_ds_cfg=None,
                suffix=args.suffix,
                num_workers=args.num_workers,
                max_natoms=args.max_natoms,
            )
        print(f"[OK] predictions written under: {base}")
    else:
        trainer = Trainer(
            config=model.config,
            default_root_dir=str(ckpt_dir),
            use_distributed_sampler=False,
            accelerator="gpu" if device.type == "cuda" else "cpu",
            devices=1,
            callbacks=[],
        )
        ran = False
        if args.val_src is not None:
            trainer.validate(model)
            ran = True
        if args.test_src is not None:
            trainer.test(model)
            ran = True
        if not ran:
            trainer.validate(model)
            trainer.test(model)
        out_root = Path(model._get_results_dir(trainer))
        print(f"[OK] predictions written under: {out_root}")


if __name__ == "__main__":
    main()
