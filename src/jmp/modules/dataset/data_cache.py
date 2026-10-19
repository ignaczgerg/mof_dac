import hashlib
import json
import os
import socket
from logging import getLogger
from pathlib import Path

import numpy as np

log = getLogger(__name__)


def _unique_suffix() -> str:
    return f"{socket.gethostname()}.{os.getpid()}.{os.urandom(4).hex()}"

FILTER_VERSION = "1"
NORM_VERSION = "1"
SIZES_VERSION = "1"
STATS_VERSION = "1"
_DISABLE_ENV = "JMP_DISABLE_DATA_CACHE"


def disabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def cache_root(logging_path: str | None) -> Path:
    base = logging_path or os.environ.get("FT_LOGGING") or os.getcwd()
    return Path(base) / "data_cache"


def _stat_signature(path) -> str:
    if path is None:
        return ""
    path = Path(path)
    if not path.exists():
        return ""
    if path.is_dir():
        files = [f for f in sorted(path.rglob("*")) if f.is_file()]
    else:
        files = [path]
    parts = []
    for f in files:
        st = f.stat()
        parts.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(parts)


def _hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _atomic_write_bytes(target: Path, write_fn) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{_unique_suffix()}")
    with open(tmp, "wb") as fh:
        write_fn(fh)
    os.replace(tmp, target)


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, target)


def filter_cache_key(src, metadata_path, split, max_natoms) -> str:
    return _hash(
        {
            "kind": "filter",
            "version": FILTER_VERSION,
            "src": str(src),
            "src_sig": _stat_signature(src),
            "metadata_sig": _stat_signature(metadata_path),
            "split": split,
            "max_natoms": max_natoms,
        }
    )


def load_filtered_indices(root: Path, key: str) -> list[int] | None:
    f = root / key / "filtered_indices.npy"
    if not f.is_file():
        return None
    try:
        return np.load(f).astype(np.int64).tolist()
    except Exception as e:
        log.warning("data_cache: failed to load filtered indices from %s (%s)", f, e)
        return None


def save_filtered_indices(root: Path, key: str, indices: list[int], meta: dict) -> None:
    d = root / key
    arr = np.asarray(indices, dtype=np.int64)
    _atomic_write_bytes(d / "filtered_indices.npy", lambda fh: np.save(fh, arr))
    _atomic_write_text(d / "key.json", json.dumps(meta, indent=2, default=str))


def sizes_cache_key(parts, split, seed) -> str:
    return _hash(
        {
            "kind": "sizes",
            "version": SIZES_VERSION,
            "parts": list(parts),
            "split": split,
            "seed": seed,
        }
    )


def load_sizes(root: Path, key: str) -> list[int] | None:
    f = root / key / "sizes.npy"
    if not f.is_file():
        return None
    try:
        return np.load(f).astype(np.int64).tolist()
    except Exception as e:
        log.warning("data_cache: failed to load sizes from %s (%s)", f, e)
        return None


def save_sizes(root: Path, key: str, sizes: list[int], meta: dict) -> None:
    d = root / key
    arr = np.asarray(sizes, dtype=np.int64)
    _atomic_write_bytes(d / "sizes.npy", lambda fh: np.save(fh, arr))
    _atomic_write_text(d / "sizes_key.json", json.dumps(meta, indent=2, default=str))


def load_stats(root: Path, key: str) -> dict | None:
    f = root / key / "stats.json"
    if not f.is_file():
        return None
    try:
        with open(f) as fh:
            raw = json.load(fh)
        return {
            "avg_num_nodes": float(raw["avg_num_nodes"]),
            "avg_degree": float(raw["avg_degree"]),
        }
    except Exception as e:
        log.warning("data_cache: failed to load stats from %s (%s)", f, e)
        return None


def save_stats(root: Path, key: str, stats: dict, meta: dict) -> None:
    d = root / key
    payload = {
        "avg_num_nodes": float(stats["avg_num_nodes"]),
        "avg_degree": float(stats["avg_degree"]),
        **{k: v for k, v in meta.items()},
    }
    _atomic_write_text(d / "stats.json", json.dumps(payload, indent=2, default=str))


def norm_cache_key(
    filter_key, task_name, targets, norm_types, uptake_mode, heteroscedastic, sample_n
) -> str:
    return _hash(
        {
            "kind": "norm",
            "version": NORM_VERSION,
            "filter_key": filter_key,
            "task": task_name,
            "targets": list(targets),
            "norm_types": norm_types,
            "uptake_mode": uptake_mode,
            "heteroscedastic": bool(heteroscedastic),
            "sample_n": sample_n,
        }
    )


def load_normalization(root: Path, key: str):
    f = root / key / "normalization.json"
    if not f.is_file():
        return None
    from jmp.modules.transforms.normalize import NormalizationConfig as NC

    try:
        with open(f) as fh:
            raw = json.load(fh)
        return {
            name: NC(
                mean=float(v["mean"]),
                std=float(v["std"]),
                normalization_type=v.get("normalization_type", "standard"),
            )
            for name, v in raw.items()
        }
    except Exception as e:
        log.warning("data_cache: failed to load normalization from %s (%s)", f, e)
        return None


def save_normalization(root: Path, key: str, normalization: dict, meta: dict) -> None:
    serializable = {}
    for name, nc in normalization.items():
        mean = getattr(nc, "mean", None)
        std = getattr(nc, "std", None)
        if mean is None or std is None:
            continue
        serializable[name] = {
            "mean": float(mean),
            "std": float(std),
            "normalization_type": getattr(nc, "normalization_type", "standard"),
        }
    d = root / key
    _atomic_write_text(d / "normalization.json", json.dumps(serializable, indent=2))
    _atomic_write_text(d / "key.json", json.dumps(meta, indent=2, default=str))
