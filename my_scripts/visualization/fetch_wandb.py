import os
import wandb
import pandas as pd

def fetch_runs(entity: str, project: str):
    api = wandb.Api()
    return list(api.runs(f"{entity}/{project}"))

def flatten_dict(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat[f"{prefix}{k}_{subk}"] = subv
        else:
            flat[f"{prefix}{k}"] = v
    return flat

def fetch_wandb_summary(
    entity: str,
    project: str,
    use_history: bool = False,
    # val_metric: str = "m_val_loss",
):
    runs = fetch_runs(entity, project)
    records = []
    for run in runs:
        rec = {
            "run_id":  run.id,
            "name":    run.name,
            "created": run.created_at,
        }
        rec.update(flatten_dict(run.config,  prefix="cfg_"))
        rec.update(flatten_dict(run.summary, prefix="sum_"))
        records.append(rec)
    df_summary = pd.DataFrame.from_records(records)

    if not use_history:
        return df_summary

    histories = []
    for run in runs:
        hist = run.history(samples=-1)
        if hist.empty:
            continue
        hist = hist.add_prefix("m_")
        hist.insert(0, "run_id", run.id)
        histories.append(hist)

    df_history = pd.concat(histories, ignore_index=True, sort=False) if histories else pd.DataFrame()
    if val_metric not in df_history.columns:
        raise KeyError(f"Validation metric '{val_metric}' not in history columns")

    best_idx = df_history.groupby("run_id")[val_metric].idxmin()
    df_best = df_history.loc[best_idx].reset_index(drop=True)

    return df_summary, df_history, df_best

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Fetch W&B runs summary (+ optional histories/beststep)"
    )
    parser.add_argument("--entity",     default="theme4")
    parser.add_argument("--project",    default="aramco_dac")
    parser.add_argument("--use_history", action="store_true")
    parser.add_argument("--out-summary", default="runs_summary.csv")
    parser.add_argument("--out-history", default="runs_history.csv")
    parser.add_argument("--out-best",    default="runs_best_metrics.csv")
    parser.add_argument("--val-metric", default="m_val_loss")
    args = parser.parse_args()

    result = fetch_wandb_summary(
        entity=args.entity,
        project=args.project,
        use_history=args.use_history,
        val_metric=args.val_metric,
    )

    if args.use_history:
        df_summary, df_history, df_best = result
        df_history.to_csv(args.out_history, index=False)
        df_best.to_csv(args.out_best,    index=False)
        print(f"Saved histories to {args.out_history} and best metrics to {args.out_best}")
    else:
        df_summary = result
    df_summary.to_csv(args.out_summary, index=False)
    print(f"Saved summary to {args.out_summary}")