import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import torch
from scipy.integrate._lebedev import lebedev_rule
from scipy.interpolate import RegularGridInterpolator

LOGGER = logging.getLogger("co2-density-preprocess")

ZBL_COEFFS = [
    (0.1818, 3.2),
    (0.5099, 0.9423),
    (0.2802, 0.4029),
    (0.02817, 0.2016),
]


def zbl_screening(r: np.ndarray, a: float = 1.5) -> np.ndarray:
    x = r / a
    phi = np.zeros_like(r)
    for c, d in ZBL_COEFFS:
        phi += c * np.exp(-d * x)
    return phi


def real_spherical_harmonics(lmax: int, xyz: np.ndarray) -> np.ndarray:
    """Compute real spherical harmonics for unit vectors.

    Args:
        lmax: maximum degree
        xyz: (3, N) array of unit vectors on the sphere

    Returns:
        (N, n_sh) array where n_sh = (lmax+1)^2
    """
    from e3nn.o3 import spherical_harmonics as e3nn_sh

    xyz_torch = torch.from_numpy(xyz.T).float()
    ls = list(range(lmax + 1))
    Y = e3nn_sh(ls, xyz_torch, normalize=True)
    return Y.numpy()


def parse_vtk(vtk_path: str) -> Dict:
    with open(vtk_path, "r") as f:
        header_line = f.readline()
        cell_line = f.readline().strip()
        parts = cell_line.split()
        cell_params = {
            "a": float(parts[1]),
            "b": float(parts[2]),
            "c": float(parts[3]),
            "alpha": float(parts[4]),
            "beta": float(parts[5]),
            "gamma": float(parts[6]),
        }

        f.readline()  # ASCII
        f.readline()  # DATASET STRUCTURED_POINTS

        dim_line = f.readline().strip().split()
        nx, ny, nz = int(dim_line[1]), int(dim_line[2]), int(dim_line[3])

        f.readline()  # ORIGIN
        f.readline()  # SPACING

        point_data_line = f.readline().strip().split()
        n_points = int(point_data_line[1])

        scalars = {}
        for _ in range(2):
            scalar_line = f.readline().strip()
            scalar_name = scalar_line.split()[1]
            f.readline()  # LOOKUP_TABLE default

            values = np.empty(n_points, dtype=np.float64)
            for i in range(n_points):
                values[i] = float(f.readline().strip())

            scalars[scalar_name] = values.reshape((nx, ny, nz), order="C")

    spacing_parts: List[str] = []
    with open(vtk_path, "r") as f:
        for line in f:
            if line.startswith("SPACING"):
                spacing_parts = line.strip().split()
                break
    sx = float(spacing_parts[1])
    sy = float(spacing_parts[2])
    sz = float(spacing_parts[3])

    return {
        "cell_params": cell_params,
        "dimensions": (nx, ny, nz),
        "spacing": (sx, sy, sz),
        "scalars": scalars,
    }


def build_grid_interpolator(
    vtk_data: Dict,
) -> RegularGridInterpolator:
    nx, ny, nz = vtk_data["dimensions"]
    sx, sy, sz = vtk_data["spacing"]

    density = np.zeros((nx, ny, nz), dtype=np.float64)
    for name, grid in vtk_data["scalars"].items():
        density += grid

    x_coords = np.arange(nx) * sx
    y_coords = np.arange(ny) * sy
    z_coords = np.arange(nz) * sz

    interp = RegularGridInterpolator(
        (x_coords, y_coords, z_coords),
        density,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return interp


def cell_params_to_matrix(a, b, c, alpha, beta, gamma):
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)

    cos_a = np.cos(alpha_r)
    cos_b = np.cos(beta_r)
    cos_g = np.cos(gamma_r)
    sin_g = np.sin(gamma_r)

    v2 = b * cos_g
    v3 = b * sin_g
    w2 = c * cos_b
    w3 = (c * (cos_a - cos_b * cos_g)) / sin_g
    w4 = np.sqrt(max(c * c - w2 * w2 - w3 * w3, 0.0))

    return np.array([
        [a, 0.0, 0.0],
        [v2, v3, 0.0],
        [w2, w3, w4],
    ])


def wrap_to_cell(points: np.ndarray, cell_matrix: np.ndarray) -> np.ndarray:
    inv_cell = np.linalg.inv(cell_matrix)
    frac = points @ inv_cell.T
    frac = frac % 1.0
    return frac @ cell_matrix.T


def compute_atom_sh_coefficients(
    atom_pos: np.ndarray,
    interp: RegularGridInterpolator,
    cell_matrix: np.ndarray,
    lebedev_pts: np.ndarray,
    lebedev_weights: np.ndarray,
    Y_lm: np.ndarray,
    lmax: int,
    r_cut: float,
    n_radial: int,
    zbl_a: float,
) -> np.ndarray:
    n_sh = (lmax + 1) ** 2
    n_quad = lebedev_pts.shape[1]

    radii = np.linspace(0.5, r_cut, n_radial)
    dr = radii[1] - radii[0] if n_radial > 1 else r_cut

    decay = zbl_screening(radii, a=zbl_a)

    integrated = np.zeros(n_quad)

    for ri, r in enumerate(radii):
        sphere_points = atom_pos[None, :] + r * lebedev_pts.T
        sphere_points = wrap_to_cell(sphere_points, cell_matrix)
        rho = interp(sphere_points)
        integrated += rho * decay[ri] * r * r * dr

    coeffs = np.zeros(n_sh)
    for i in range(n_sh):
        coeffs[i] = np.sum(lebedev_weights * integrated * Y_lm[:, i])

    return coeffs


def process_single_entry(
    data,
    vtk_lookup: Dict[str, str],
    lebedev_pts: np.ndarray,
    lebedev_weights: np.ndarray,
    Y_lm: np.ndarray,
    lmax: int,
    r_cut: float,
    n_radial: int,
    zbl_a: float,
):
    sid = data.sid if hasattr(data, "sid") else (data.id if hasattr(data, "id") else None)
    if sid is None:
        data.has_co2_sh = torch.tensor(False)
        n_sh = (lmax + 1) ** 2
        natoms = int(data.natoms) if hasattr(data, "natoms") else data.pos.shape[0]
        data.co2_sh_coeffs = torch.zeros(natoms, n_sh, dtype=torch.float32)
        return data

    sid_str = str(sid).strip()
    vtk_path = vtk_lookup.get(sid_str) or vtk_lookup.get(sid_str.lower())

    natoms = int(data.natoms) if hasattr(data, "natoms") else data.pos.shape[0]
    n_sh = (lmax + 1) ** 2

    if vtk_path is None:
        data.has_co2_sh = torch.tensor(False)
        data.co2_sh_coeffs = torch.zeros(natoms, n_sh, dtype=torch.float32)
        return data

    try:
        vtk_data = parse_vtk(vtk_path)
    except Exception as e:
        LOGGER.warning("Failed to parse VTK for %s: %s", sid_str, e)
        data.has_co2_sh = torch.tensor(False)
        data.co2_sh_coeffs = torch.zeros(natoms, n_sh, dtype=torch.float32)
        return data

    interp = build_grid_interpolator(vtk_data)

    cp = vtk_data["cell_params"]
    cell_matrix = cell_params_to_matrix(cp["a"], cp["b"], cp["c"], cp["alpha"], cp["beta"], cp["gamma"])

    pos = data.pos.numpy() if isinstance(data.pos, torch.Tensor) else np.array(data.pos)
    all_coeffs = np.zeros((natoms, n_sh), dtype=np.float32)

    for ai in range(natoms):
        all_coeffs[ai] = compute_atom_sh_coefficients(
            atom_pos=pos[ai],
            interp=interp,
            cell_matrix=cell_matrix,
            lebedev_pts=lebedev_pts,
            lebedev_weights=lebedev_weights,
            Y_lm=Y_lm,
            lmax=lmax,
            r_cut=r_cut,
            n_radial=n_radial,
            zbl_a=zbl_a,
        )

    data.co2_sh_coeffs = torch.tensor(all_coeffs, dtype=torch.float32)
    data.has_co2_sh = torch.tensor(True)
    return data


def build_vtk_lookup(voxels_dir: str) -> Dict[str, str]:
    lookup = {}
    voxels_path = Path(voxels_dir)
    for entry in voxels_path.iterdir():
        if entry.is_dir():
            vtk_file = entry / "DensityProfile_CO2.vtk"
            if vtk_file.exists():
                lookup[entry.name] = str(vtk_file)
                lookup[entry.name.lower()] = str(vtk_file)
    return lookup


def process_lmdb_split(
    src_lmdb_path: str,
    dst_lmdb_dir: str,
    split_name: str,
    vtk_lookup: Dict[str, str],
    lmax: int,
    r_cut: float,
    n_radial: int,
    zbl_a: float,
    lebedev_order: int,
    map_size_gb: float,
    commit_interval: int,
    log_every: int,
):
    lebedev_pts, lebedev_weights = lebedev_rule(lebedev_order)
    Y_lm = real_spherical_harmonics(lmax, lebedev_pts)

    src_path = Path(src_lmdb_path)
    if src_path.is_dir():
        lmdb_files = sorted(src_path.glob("*.lmdb"))
        if not lmdb_files:
            LOGGER.error("No .lmdb files found in %s", src_path)
            return {}
    else:
        lmdb_files = [src_path]

    dst_dir = Path(dst_lmdb_dir) / split_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_lmdb_path = dst_dir / f"{split_name}.lmdb"

    dst_env = lmdb.open(
        str(dst_lmdb_path),
        map_size=int(map_size_gb * (1024 ** 3)),
        subdir=False,
        meminit=False,
        map_async=True,
        lock=True,
        readahead=False,
    )

    global_idx = 0
    natoms_list = []
    n_with_sh = 0
    total_atoms_with_sh = 0
    n_sh = (lmax + 1) ** 2
    sh_running_sum = np.zeros(n_sh, dtype=np.float64)
    sh_running_sq_sum = np.zeros(n_sh, dtype=np.float64)

    txn = dst_env.begin(write=True)

    for lmdb_file in lmdb_files:
        src_env = lmdb.open(str(lmdb_file), readonly=True, lock=False, readahead=True)
        src_txn = src_env.begin()

        length_raw = src_txn.get(b"length")
        if length_raw is None:
            cursor = src_txn.cursor()
            n_entries = sum(1 for _ in cursor) - (1 if src_txn.get(b"length") else 0)
        else:
            n_entries = pickle.loads(length_raw)

        LOGGER.info("Processing %s (%d entries)", lmdb_file, n_entries)

        for i in range(n_entries):
            key = f"{i}".encode("ascii")
            raw = src_txn.get(key)
            if raw is None:
                LOGGER.warning("Missing key %d in %s", i, lmdb_file)
                continue

            data = pickle.loads(raw)

            data = process_single_entry(
                data,
                vtk_lookup=vtk_lookup,
                lebedev_pts=lebedev_pts,
                lebedev_weights=lebedev_weights,
                Y_lm=Y_lm,
                lmax=lmax,
                r_cut=r_cut,
                n_radial=n_radial,
                zbl_a=zbl_a,
            )

            if data.has_co2_sh.item():
                n_with_sh += 1
                coeffs_np = data.co2_sh_coeffs.numpy()
                atom_sum = coeffs_np.sum(axis=0)
                atom_sq_sum = (coeffs_np ** 2).sum(axis=0)
                n_atoms_here = coeffs_np.shape[0]
                sh_running_sum += atom_sum
                sh_running_sq_sum += atom_sq_sum
                total_atoms_with_sh += n_atoms_here

            out_key = f"{global_idx}".encode("ascii")
            txn.put(out_key, pickle.dumps(data, protocol=-1))
            natoms_list.append(int(data.natoms) if hasattr(data, "natoms") else data.pos.shape[0])
            global_idx += 1

            if global_idx % commit_interval == 0:
                txn.commit()
                txn = dst_env.begin(write=True)

            if global_idx % log_every == 0:
                LOGGER.info("  processed %d entries (%d with SH)", global_idx, n_with_sh)

        src_env.close()

    txn.put(b"length", pickle.dumps(global_idx, protocol=-1))
    txn.commit()
    dst_env.sync()
    dst_env.close()

    meta_path = dst_dir / "metadata.npz"
    np.savez(meta_path, natoms=np.array(natoms_list, dtype=np.int32))

    LOGGER.info(
        "Finished %s: %d total entries, %d with CO2 SH data",
        split_name, global_idx, n_with_sh,
    )

    stats = {}
    if total_atoms_with_sh > 0:
        mean = sh_running_sum / total_atoms_with_sh
        var = sh_running_sq_sum / total_atoms_with_sh - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-12))
        stats = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "n_atoms": int(total_atoms_with_sh),
            "n_structures": int(n_with_sh),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CO2 density VTK files into per-atom SH coefficients and write new LMDBs"
    )
    parser.add_argument("--src_lmdb_root", type=str, required=True,
                        help="Root of source LMDB dataset (contains train/val/test subdirs)")
    parser.add_argument("--voxels_dir", type=str, required=True,
                        help="Directory containing MOF voxel subdirs with DensityProfile_CO2.vtk")
    parser.add_argument("--dst_lmdb_root", type=str, required=True,
                        help="Root for output LMDB dataset")
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--r_cut", type=float, default=6.0)
    parser.add_argument("--n_radial", type=int, default=20)
    parser.add_argument("--zbl_a", type=float, default=1.5,
                        help="ZBL screening length parameter (Angstroms)")
    parser.add_argument("--lebedev_order", type=int, default=29,
                        help="Lebedev quadrature order (29 -> 302 points)")
    parser.add_argument("--map_size_gb", type=float, default=16.0)
    parser.add_argument("--commit_interval", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    LOGGER.info("Building VTK lookup from %s ...", args.voxels_dir)
    vtk_lookup = build_vtk_lookup(args.voxels_dir)
    LOGGER.info("Found %d MOFs with VTK density data", len(vtk_lookup) // 2)

    dst_root = Path(args.dst_lmdb_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in args.splits:
        src_split = Path(args.src_lmdb_root) / split
        if not src_split.exists():
            LOGGER.warning("Source split %s not found, skipping", src_split)
            continue

        stats = process_lmdb_split(
            src_lmdb_path=str(src_split),
            dst_lmdb_dir=str(dst_root),
            split_name=split,
            vtk_lookup=vtk_lookup,
            lmax=args.lmax,
            r_cut=args.r_cut,
            n_radial=args.n_radial,
            zbl_a=args.zbl_a,
            lebedev_order=args.lebedev_order,
            map_size_gb=args.map_size_gb,
            commit_interval=args.commit_interval,
            log_every=args.log_every,
        )
        all_stats[split] = stats

    stats_path = dst_root / "normalization_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    LOGGER.info("Normalization stats written to %s", stats_path)

    if "train" in all_stats and all_stats["train"]:
        ts = all_stats["train"]
        LOGGER.info("Train set stats: %d structures, %d atoms with SH", ts["n_structures"], ts["n_atoms"])
        LOGGER.info("  mean (first 5): %s", ts["mean"][:5])
        LOGGER.info("  std  (first 5): %s", ts["std"][:5])

    # Copy split_keys and splits.json if they exist
    src_root = Path(args.src_lmdb_root)
    for fname in ["splits.json"]:
        src_file = src_root / fname
        if src_file.exists():
            import shutil
            shutil.copy2(str(src_file), str(dst_root / fname))
            LOGGER.info("Copied %s", fname)

    split_keys_src = src_root / "split_keys"
    split_keys_dst = dst_root / "split_keys"
    if split_keys_src.exists() and not split_keys_dst.exists():
        import shutil
        shutil.copytree(str(split_keys_src), str(split_keys_dst))
        LOGGER.info("Copied split_keys directory")

    LOGGER.info("Done. Output at %s", dst_root)


if __name__ == "__main__":
    main()
