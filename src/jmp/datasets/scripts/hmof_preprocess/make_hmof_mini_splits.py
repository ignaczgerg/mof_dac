import argparse, os, pickle
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedShuffleSplit
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


PROPS = [
    "CO2_uptake_P0.15bar_T298K [mmol/g]",
    "heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]",
    "excess_CO2_uptake_P0.15bar_T298K [mmol/g]",
    "CO2_binary_uptake_P0.15bar_T298K [mmol/g]",
    "heat_adsorption_CO2_binary_P0.15bar_T298K [kcal/mol]",
    "excess_CO2_binary_uptake_P0.15bar_T298K [mmol/g]",
    "N2_binary_uptake_P0.85bar_T298K [mmol/g]",
    "heat_adsorption_N2_binary_P0.85bar_T298K [kcal/mol]",
    "excess_N2_binary_uptake_P0.85bar_T298K [mmol/g]",
    "CO2/N2_selectivity",
]

def _load_df(csv_path: Path) -> pd.DataFrame:
    usecols = ["id"] + PROPS
    df = pd.read_csv(csv_path, usecols=usecols)
    for c in PROPS:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.fillna(s.median())
        df[c] = s
    return df

def _build_strata(df: pd.DataFrame, approx_bins: int = 50) -> np.ndarray:
    """
    Build a single stratification label by:
      1) standardizing target columns
      2) taking PC1
      3) quantile-binning PC1
    We adaptively reduce the number of bins until all global bins have >= 2 items.
    """
    X = df[PROPS].to_numpy(dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    Z = (X - mu) / sd

    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    pc1 = U[:, 0] * S[0]

    # try requested approx_bins first, then back off
    backoff = [approx_bins, 64, 50, 40, 32, 24, 20, 16, 12, 10, 8, 6, 5, 4, 3, 2]
    seen = set()
    candidates = [b for b in backoff if not (b in seen or seen.add(b))]

    ser = pd.Series(pc1)
    ranks = ser.rank(method="first")
    for b in candidates:
        try:
            q = pd.qcut(ranks, q=b, labels=False, duplicates="drop")
            cnt = np.bincount(q.astype(int))
            if cnt.min() >= 2:
                return q.to_numpy(dtype=int)
        except Exception:
            continue
    q = pd.qcut(ranks, q=2, labels=False, duplicates="drop")
    return q.to_numpy(dtype=int)


def _merge_rare_bins(y: np.ndarray, min_count: int = 2) -> np.ndarray:
    """
    If some classes in y have < min_count samples, merge them into the nearest
    (by label index) class that satisfies the count. Returns a new y.
    """
    y = np.asarray(y, dtype=int).copy()
    counts = np.bincount(y)
    rare = np.where(counts < min_count)[0]
    if rare.size == 0:
        return y
    valid = np.where(counts >= min_count)[0]
    # If everything is rare, collapse to a single class
    if valid.size == 0:
        return np.zeros_like(y)
    for r in rare:
        nearest = valid[np.argmin(np.abs(valid - r))]
        y[y == r] = nearest
    return y


def _stratified_pick(idx, y, n_pick, seed=42):
    idx = np.asarray(idx, dtype=int)
    y_sub = np.asarray(y[idx], dtype=int)
    if n_pick >= len(idx):
        return idx.tolist(), np.array([], dtype=int).tolist()

    # Try scikit-learn first, after fixing rare bins in THIS subset
    if _HAVE_SK:
        y_fixed = _merge_rare_bins(y_sub, min_count=2)
        try:
            test_size = n_pick / len(idx)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_i, picked_i = next(sss.split(idx, y_fixed))
            return idx[picked_i].tolist(), idx[train_i].tolist()
        except ValueError:
            # fall through to manual proportional sampler
            pass

    # Fallback: proportional sampling per (possibly merged) stratum
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(y_sub, return_counts=True)
    props = counts / counts.sum()
    target = (props * n_pick).astype(int)
    while target.sum() < n_pick:
        target[np.argmax(props - target / max(1, n_pick))] += 1

    picked = []
    pool = []
    for u, t in zip(uniq, target):
        pool_u = idx[y_sub == u]
        if len(pool_u) <= t:
            picked.extend(pool_u.tolist())
        else:
            picked.extend(rng.choice(pool_u, size=t, replace=False).tolist())
        left_u = [x for x in pool_u if x not in picked]
        pool.extend(left_u)

    if len(picked) < n_pick and pool:
        need = n_pick - len(picked)
        picked.extend(rng.choice(pool, size=need, replace=False).tolist())

    rest = [x for x in idx.tolist() if x not in picked]
    return picked, rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--out_dir", required=True, help="Where to save split_keys_mini/*.pkl")
    ap.add_argument("--train", type=int, default=10_000)
    ap.add_argument("--val",   type=int, default=1_000)
    ap.add_argument("--test",  type=int, default=1_000)
    ap.add_argument("--bins",  type=int, default=8)
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()

    csv_path = Path(args.dataset_path) / "raw" / "all_MOFs_screening_data_cleaned.csv"
    df = _load_df(csv_path)

    base_idx = np.arange(len(df), dtype=int)
    y = _build_strata(df, approx_bins=args.bins)

    total_needed = args.train + args.val + args.test
    picked12k, restA = _stratified_pick(base_idx, y, total_needed, seed=args.seed)

    picked2k, train10k = _stratified_pick(picked12k, y, args.val + args.test, seed=args.seed + 1)

    val1k, test1k = _stratified_pick(picked2k, y, args.val, seed=args.seed + 2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_keys_mini").mkdir(parents=True, exist_ok=True)

    with open(out_dir / "split_keys_mini" / "train_keys.pkl", "wb") as f:
        pickle.dump([int(x) for x in train10k], f)
    with open(out_dir / "split_keys_mini" / "val_keys.pkl", "wb") as f:
        pickle.dump([int(x) for x in val1k], f)
    with open(out_dir / "split_keys_mini" / "test_keys.pkl", "wb") as f:
        pickle.dump([int(x) for x in test1k], f)

    print("Saved:")
    print("  train:", len(train10k))
    print("  val:  ", len(val1k))
    print("  test: ", len(test1k))

if __name__ == "__main__":
    main()
