from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

DEFAULT_OUT_ROOT = Path("/ibex/project/c2261/dac_iclr/finetune/ablation_eval")
MANIFEST = Path(__file__).resolve().parent / "ablation_runs.tsv"

TARGET_LABELS = {
    "qst_co2": ("$Q_{st}$(CO$_2$) [kJ/mol]", "qst_co2 (kJ/mol)"),
    "co2_uptake": ("CO$_2$ uptake [mmol/g]", "co2_uptake (mmol/g)"),
}

METRIC_COLUMNS = [
    ("r2", "$R^2$"),
    ("pearson", "Pearson $r$"),
    ("spearman", "Spearman $\\rho$"),
    ("rmse", "RMSE"),
    ("mae", "MAE"),
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            run_id, run_tag, block, label = line.split("\t")
            rows.append({"run_id": run_id, "run_tag": run_tag, "block": block, "label": label})
    return rows


def load_predictions(pred_root: Path, split: str, target: str):
    files = sorted(pred_root.glob(f"final/{split}/*/{target}.npz"))
    if not files:
        return None
    sid, gt, pred = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        sid.append(np.asarray(d["sid"], dtype=object))
        gt.append(np.asarray(d["gt"], dtype=np.float64))
        pred.append(np.asarray(d["pred"], dtype=np.float64))
    sid = np.concatenate(sid)
    gt = np.concatenate(gt)
    pred = np.concatenate(pred)

    _, keep = np.unique(sid.astype(str), return_index=True)
    keep = np.sort(keep)
    sid, gt, pred = sid[keep], gt[keep], pred[keep]

    finite = np.isfinite(gt) & np.isfinite(pred)
    return sid[finite], gt[finite], pred[finite], int(sid.size)


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    n = int(gt.size)
    if n < 2:
        return {"n": n, "r2": np.nan, "pearson": np.nan, "spearman": np.nan,
                "rmse": np.nan, "mae": np.nan}
    err = pred - gt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    pearson = float(stats.pearsonr(gt, pred)[0]) if np.std(gt) > 0 and np.std(pred) > 0 else np.nan
    spearman = float(stats.spearmanr(gt, pred)[0])
    return {
        "n": n,
        "r2": float(r2),
        "pearson": pearson,
        "spearman": spearman,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }


def fmt(value: float, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def column_digits(rows, key, digits):
    vals = [abs(r["metrics"][key]) for r in rows
            if r["metrics"] and np.isfinite(r["metrics"].get(key, np.nan)) and r["metrics"][key] != 0]
    if not vals:
        return digits
    smallest = min(vals)
    if smallest >= 0.1:
        return digits
    extra = int(np.ceil(-np.log10(smallest))) - 1
    return int(min(digits + extra, 6))


def latex_table(rows, target, split, digits, note_missing, note_nonfinite):
    header = " & ".join(["Run", "Setting"] + [h for _, h in METRIC_COLUMNS]) + " \\\\"
    lines = [
        f"% {TARGET_LABELS.get(target, (target, target))[1]} -- {split} split",
        "\\begin{tabular}{ll" + "r" * len(METRIC_COLUMNS) + "}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    per_column = {key: column_digits(rows, key, digits) for key, _ in METRIC_COLUMNS}
    prev_block = None
    for row in rows:
        if prev_block is not None and row["block"] != prev_block:
            lines.append("\\midrule")
        prev_block = row["block"]
        m = row["metrics"]
        cells = [row["run_id"], row["label"]]
        for key, _ in METRIC_COLUMNS:
            cells.append(fmt(m.get(key) if m else None, per_column[key]))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if note_missing:
        lines.append("% no predictions exported: " + ", ".join(note_missing))
    if note_nonfinite:
        lines.append("% predictions are non-finite (diverged run): " + ", ".join(note_nonfinite))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser("Aggregate ablation predictions into a LaTeX metrics table")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--targets", nargs="+", default=["qst_co2", "co2_uptake"])
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--report", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--predictions_csv", default=None)
    ap.add_argument("--no_predictions_csv", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    runs = read_manifest(Path(args.manifest))

    report_path = Path(args.report) if args.report else out_root / f"ablation_metrics_{args.split}.txt"
    csv_path = Path(args.csv) if args.csv else out_root / f"ablation_metrics_{args.split}.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {t: [] for t in args.targets}
    missing: dict[str, list[str]] = {t: [] for t in args.targets}
    nonfinite: dict[str, list[str]] = {t: [] for t in args.targets}
    csv_rows = []
    pred_frames = []

    for run in runs:
        pred_root = out_root / run["run_tag"] / "predictions"
        for target in args.targets:
            loaded = load_predictions(pred_root, args.split, target)
            if loaded is None:
                missing[target].append(run["run_id"])
                results[target].append({**run, "metrics": None})
                continue
            sid, gt, pred, n_raw = loaded
            if gt.size == 0:
                nonfinite[target].append(run["run_id"])
            metrics = compute_metrics(gt, pred)
            metrics["n_exported"] = n_raw
            pred_frames.append((run["run_id"], run["run_tag"], target, sid, gt, pred))
            results[target].append({**run, "metrics": metrics})
            csv_rows.append({
                "run_id": run["run_id"], "run_tag": run["run_tag"], "block": run["block"],
                "label": run["label"], "target": target, "split": args.split, **metrics,
            })

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["run_id", "run_tag", "block", "label", "target", "split",
                        "n", "n_exported", "r2", "pearson", "spearman", "rmse", "mae"],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    if pred_frames and not args.no_predictions_csv:
        preds_path = (Path(args.predictions_csv) if args.predictions_csv
                      else out_root / f"ablation_predictions_{args.split}.csv.gz")
        import gzip
        with gzip.open(preds_path, "wt", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["run_id", "run_tag", "target", "split", "sid", "gt", "pred"])
            for run_id, run_tag, target, sid, gt, pred in pred_frames:
                for s_, g_, p_ in zip(sid.tolist(), gt.tolist(), pred.tolist()):
                    writer.writerow([run_id, run_tag, target, args.split, s_, g_, p_])
        print(f"[ok] wrote {preds_path}")

    blocks = []
    for target in args.targets:
        blocks.append(latex_table(results[target], target, args.split, args.digits,
                                  missing[target], nonfinite[target]))
    text = "\n\n".join(blocks) + "\n"
    report_path.write_text(text)

    print(text)
    print(f"[ok] wrote {report_path}")
    print(f"[ok] wrote {csv_path}")
    for target in args.targets:
        if missing[target]:
            print(f"[warn] {target}: no {args.split} predictions for {', '.join(missing[target])}")
        if nonfinite[target]:
            print(f"[warn] {target}: all predictions non-finite for {', '.join(nonfinite[target])}")


if __name__ == "__main__":
    main()
