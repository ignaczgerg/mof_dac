import argparse
import os
from pathlib import Path
import pickle
import lmdb
import numpy as np
import torch
import pickle, lmdb, numpy as np
from pathlib import Path

LABEL_KEYS = [
    "qst_co2",
    "qst_h2o",
    # "qst_n2",
    "kh_co2",
    "kh_h2o",
    # "kh_n2",
    # "selectivity_co2_n2",
    "selectivity_co2_h2o",
    "co2_uptake",
    # "n2_uptake",
]

SRC_ROOTS = {
    "train": "/ibex/project/c2261/datasets/adsorption-mof-db2/lmdb/train",
    "val":   "/ibex/project/c2261/datasets/adsorption-mof-db2/lmdb/val",
    "test":  "/ibex/project/c2261/datasets/adsorption-mof-db2/lmdb/test",
}
DST_ROOTS = {
    "train": "/ibex/project/c2261/datasets/log-db2/lmdb/train",
    "val":   "/ibex/project/c2261/datasets/log-db2/lmdb/val",
    "test":  "/ibex/project/c2261/datasets/log-db2/lmdb/test",
}

def safe_log_abs(x: float, eps: float) -> float:
    return float(np.log(np.abs(float(x)) + eps))


def _to_float(v):
    import torch, numpy as np
    if isinstance(v, torch.Tensor): return float(v.item())
    if isinstance(v, np.ndarray):  return float(v.reshape(()).item())
    return float(v)


def _like(v, x: float):
    import torch, numpy as np
    if isinstance(v, torch.Tensor): return torch.tensor(x, dtype=v.dtype, device='cpu')
    if isinstance(v, np.ndarray):  return np.asarray(x, dtype=v.dtype)
    return float(x)

def _get_attr(obj, name):
    if isinstance(obj, dict): return obj[name], 'dict'
    if hasattr(obj, name):    return getattr(obj, name), 'attr'
    try:                      return obj[name], 'getitem'
    except Exception as e:    raise KeyError(name) from e

def _set_attr(obj, name, value, how):
    if how == 'dict':   obj[name] = value
    elif how == 'attr': setattr(obj, name, value)
    elif how == 'getitem': obj[name] = value
    else: raise RuntimeError("Unknown set mode")


def passes_predicate(dp) -> bool:
    try:
        kh_co2,_=_get_attr(dp,'kh_co2')
        # kh_n2,_=_get_attr(dp,'kh_n2')
        kh_h2o,_=_get_attr(dp,'kh_h2o')
        qst_co2,_=_get_attr(dp,'qst_co2') 
        qst_h2o,_=_get_attr(dp,'qst_h2o') 
        # qst_n2,_=_get_attr(dp,'qst_n2')
        # n2_uptake,_=_get_attr(dp,'n2_uptake'); 
        co2_uptake,_=_get_attr(dp,'co2_uptake')
        selectivity_co2_h2o, _ = _get_attr(dp, 'selectivity_co2_h2o')
        ###
        kh_co2=_to_float(kh_co2)
        # kh_n2=_to_float(kh_n2)
        kh_h2o=_to_float(kh_h2o)
        qst_co2=_to_float(qst_co2) 
        qst_h2o=_to_float(qst_h2o)
        # qst_n2=_to_float(qst_n2)
        # n2_uptake=_to_float(n2_uptake)
        co2_uptake=_to_float(co2_uptake)
        selectivity_co2_h2o,_=_get_attr(dp,'selectivity_co2_h2o')
    except KeyError:
        return False
    return (
        (kh_co2  < 1.0) and 
        # (kh_n2 < 1.0) and 
        (kh_h2o < 1.0) and
        (qst_co2 < 0.0) and 
        (qst_h2o < 0.0) and 
        # (qst_n2  < 0.0) and
        # (n2_uptake != 0.0) and 
        (co2_uptake > 0.0) and
        (selectivity_co2_h2o > 0.0)
    )

def compute_mapsize(src_file: Path) -> int:
    try: sz = src_file.stat().st_size
    except FileNotFoundError: sz = 1 << 30
    return max(1 << 30, int(sz * 1.5) + (1 << 26))

def iter_keys_in_order(env, prefer_metadata_keys=None):
    if prefer_metadata_keys is not None:
        for k in prefer_metadata_keys:
            yield k.encode('ascii')
        return
    with env.begin(buffers=True) as txn:
        with txn.cursor() as cur:
            for k, _ in cur:
                yield bytes(k)

def load_src_metadata(src_dir: Path):
    p = src_dir / "metadata.npz"
    if p.exists():
        return dict(np.load(p, allow_pickle=True))
    return None

def extract_keys_from_meta_for_shard(meta, shard_basename):
    if meta is None:
        return None
    candidate_fields = ['keys', 'db_keys', 'shard_keys']
    name_fields = ['db_names', 'shard_names', 'files', 'paths']
    keys_field = next((f for f in candidate_fields if f in meta), None)
    names_field = next((f for f in name_fields if f in meta), None)
    if keys_field is None:
        return None
    if names_field and len(meta[keys_field]) == len(meta[names_field]):
        names = [str(x) for x in meta[names_field].tolist()]
        try:
            idx = names.index(shard_basename)
        except ValueError:
            base_names = [Path(n).name for n in names]
            idx = base_names.index(shard_basename)
        keys_list = meta[keys_field][idx]
        if isinstance(keys_list, (list, tuple)):
            return [str(k) for k in keys_list]
        try:
            return [str(k) for k in keys_list.tolist()]
        except Exception:
            return None
    return None

def write_metadata(dst_dir: Path, shard_order, keys_by_shard, src_meta=None, include_keys: bool = True):
    out = {}
    if src_meta is not None:
        for k, v in src_meta.items():
            out[k] = v

    names = [Path(s).name for s in shard_order]
    lengths = np.array([len(keys_by_shard.get(nm, [])) for nm in names], dtype=np.int64)
    out['db_names'] = np.array(names, dtype=object)
    out['lengths']  = lengths

    if include_keys:
        keys_obj = np.empty(len(names), dtype=object)
        for i, nm in enumerate(names):
            keys_obj[i] = keys_by_shard.get(nm, [])
        out['keys'] = keys_obj
    else:
        out.pop('keys', None)

    np.savez(dst_dir / "metadata.npz", **out)

def transform_shard(src_file: Path, dst_file: Path, eps: float, commit_every: int = 20000):
    dst_file.parent.mkdir(parents=True, exist_ok=True)

    if dst_file.exists():
        dst_file.unlink()
    lock_path = dst_file.with_name(dst_file.name + "-lock")
    if lock_path.exists():
        lock_path.unlink()

    src_env = lmdb.open(str(src_file),
                        readonly=True, lock=False, subdir=False,
                        max_readers=1024, readahead=False, meminit=False)
    dst_env = lmdb.open(str(dst_file),
                        map_size=compute_mapsize(src_file),
                        subdir=False, lock=True, map_async=False, writemap=False,
                        readahead=False, meminit=False)

    kept_new_keys = []
    n_total = n_kept = 0
    new_idx = 0

    with src_env.begin(buffers=True) as s_txn:
        with s_txn.cursor() as cur:
            d_txn = dst_env.begin(write=True)
            for k_bytes, v_bytes in cur:
                n_total += 1
                try:
                    dp = pickle.loads(bytes(v_bytes))
                except Exception:
                    if (n_total % commit_every) == 0:
                        d_txn.commit(); d_txn = dst_env.begin(write=True)
                    continue

                if not passes_predicate(dp):
                    if (n_total % commit_every) == 0:
                        d_txn.commit(); d_txn = dst_env.begin(write=True)
                    continue

                try:
                    for name in LABEL_KEYS:
                        v, how = _get_attr(dp, name)
                        _set_attr(dp, name, _like(v, safe_log_abs(v, eps)), how)
                except KeyError:
                    if (n_total % commit_every) == 0:
                        d_txn.commit(); d_txn = dst_env.begin(write=True)
                    continue

                out_buf = pickle.dumps(dp, protocol=pickle.HIGHEST_PROTOCOL)
                new_key = f"{new_idx}".encode("ascii")
                d_txn.put(new_key, out_buf, overwrite=True)

                kept_new_keys.append(str(new_idx))
                new_idx += 1; n_kept += 1

                if (n_total % commit_every) == 0:
                    d_txn.commit(); d_txn = dst_env.begin(write=True)
            d_txn.commit()

    dst_env.sync(); dst_env.close(); src_env.close()
    print(f"[{src_file.name}] kept {n_kept} / {n_total}")
    return kept_new_keys



def main():
    ap = argparse.ArgumentParser(description="Filter + log(abs(.)) labels and rewrite LMDBs.")
    ap.add_argument("--eps", type=float, default=0, help="epsilon added inside log(abs(x)+eps)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train","val","test"])
    ap.add_argument("--commit-every", type=int, default=20000)
    ap.add_argument("--src-train", default=SRC_ROOTS["train"])
    ap.add_argument("--src-val",   default=SRC_ROOTS["val"])
    ap.add_argument("--src-test",  default=SRC_ROOTS["test"])
    ap.add_argument("--dst-train", default=DST_ROOTS["train"])
    ap.add_argument("--dst-val",   default=DST_ROOTS["val"])
    ap.add_argument("--dst-test",  default=DST_ROOTS["test"])
    args = ap.parse_args()

    src_map = {"train": Path(args.src_train), "val": Path(args.src_val), "test": Path(args.src_test)}
    dst_map = {"train": Path(args.dst_train), "val": Path(args.dst_val), "test": Path(args.dst_test)}

    for split in args.splits:
        src_dir = src_map[split]
        dst_dir = dst_map[split]
        dst_dir.mkdir(parents=True, exist_ok=True)

        shards = sorted(src_dir.glob("data.*.lmdb"))
        if not shards:
            print(f"[{split}] No shards found in {src_dir}")
            continue

        src_meta = load_src_metadata(src_dir)

        keys_by_shard = {}
        for sf in shards:
            df = dst_dir / sf.name
            kept = transform_shard(sf, df, eps=args.eps, commit_every=args.commit_every)
            keys_by_shard[sf.name] = kept

        write_metadata(dst_dir, shards, keys_by_shard, src_meta=src_meta, include_keys=True)
        print(f"[{split}] Done -> {dst_dir}")

if __name__ == "__main__":
    main()
