#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def parse_res_file(path: str) -> dict[str, float]:
    """Parse .res file → lcd (Di), pld (Df), gcd (Dif)."""
    with open(path) as f:
        line = f.read().strip()
    parts = line.split()
    # Format: "filename    Di Df Dif"
    if len(parts) >= 4:
        return {
            "lcd": float(parts[-3]),   # Di  — largest included sphere
            "pld": float(parts[-2]),   # Df  — largest free sphere
            "gcd": float(parts[-1]),   # Dif — included along free-sphere path
        }
    raise ValueError(f"Unexpected .res format: {line!r}")


def _extract_field(line: str, field: str) -> float:
    """Extract a named numeric field from a zeo++ summary line."""
    m = re.search(rf"{re.escape(field)}:\s*([\d.eE+\-]+)", line)
    if m is None:
        raise ValueError(f"Field {field!r} not found in: {line!r}")
    return float(m.group(1))


def parse_sa_file(path: str) -> dict[str, float]:
    """Parse .sa file → unitcell_volume, density, asa (m²/g)."""
    with open(path) as f:
        line = f.read().strip()
    return {
        "unitcell_volume": _extract_field(line, "Unitcell_volume"),
        "density": _extract_field(line, "Density"),
        "asa": _extract_field(line, "ASA_m^2/g"),
    }


def parse_vol_file(path: str) -> dict[str, float]:
    """Parse .vol file → av (cm³/g), nav (cm³/g)."""
    with open(path) as f:
        line = f.read().strip()
    return {
        "av": _extract_field(line, "AV_cm^3/g"),
        "nav": _extract_field(line, "NAV_cm^3/g"),
    }


def compute_properties(
    cif_path: Path,
    zeopp_bin: str,
    probe_radius: float = 1.86,
    num_samples: int = 5000,
) -> dict[str, float] | None:
    """Run zeo++ on a single CIF and return all 8 properties."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res_file = os.path.join(tmpdir, "out.res")
        sa_file = os.path.join(tmpdir, "out.sa")
        vol_file = os.path.join(tmpdir, "out.vol")

        cmd = [
            zeopp_bin,
            "-res", res_file,
            "-sa", str(probe_radius), str(probe_radius), str(num_samples), sa_file,
            "-vol", str(probe_radius), str(probe_radius), str(num_samples), vol_file,
            str(cif_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            print(f"[WARN] Timeout for {cif_path.name}", file=sys.stderr)
            return None

        if result.returncode != 0:
            print(
                f"[WARN] zeo++ failed for {cif_path.name}: {result.stderr.strip()}",
                file=sys.stderr,
            )
            return None

        try:
            props = {}
            props.update(parse_res_file(res_file))
            props.update(parse_sa_file(sa_file))
            props.update(parse_vol_file(vol_file))
            return props
        except Exception as e:
            print(f"[WARN] Parse error for {cif_path.name}: {e}", file=sys.stderr)
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute MOF geometric properties using zeo++ CLI."
    )
    parser.add_argument(
        "--cif_dirs", nargs="+", required=True,
        help="Directories containing CIF files.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--zeopp", type=str,
        default="/home/ignaczg/zeo++-0.3/network",
        help="Path to the zeo++ `network` binary.",
    )
    parser.add_argument(
        "--probe_radius", type=float, default=1.86,
        help="Probe radius in Angstroms (default: 1.86 for CO2).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5000,
        help="Monte Carlo samples for SA/volume (default: 5000).",
    )
    args = parser.parse_args()

    # Validate zeo++ binary
    if not os.path.isfile(args.zeopp):
        print(f"Error: zeo++ binary not found at {args.zeopp}", file=sys.stderr)
        sys.exit(1)

    # Discover CIF files
    cif_files: list[Path] = []
    for d in args.cif_dirs:
        d = Path(d)
        if not d.is_dir():
            print(f"[WARN] Skipping non-existent directory: {d}", file=sys.stderr)
            continue
        cif_files.extend(sorted(d.glob("*.cif")))

    if not cif_files:
        print("Error: no CIF files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(cif_files)} CIF files. Computing properties...")

    columns = ["id", "lcd", "pld", "gcd", "unitcell_volume", "density", "asa", "av", "nav"]
    rows: list[dict] = []
    failed: list[str] = []

    for i, cif in enumerate(cif_files, 1):
        print(f"  [{i}/{len(cif_files)}] {cif.name} ... ", end="", flush=True)
        props = compute_properties(
            cif, args.zeopp,
            probe_radius=args.probe_radius,
            num_samples=args.num_samples,
        )
        if props is None:
            print("FAILED")
            failed.append(cif.name)
            continue
        props["id"] = cif.stem
        rows.append(props)
        print(
            f"LCD={props['lcd']:.2f}  PLD={props['pld']:.2f}  "
            f"ASA={props['asa']:.2f}  AV={props['av']:.4f}"
        )

    df = pd.DataFrame(rows, columns=columns)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(rows)} entries to {args.output}")

    if failed:
        print(f"\n{len(failed)} CIFs failed:", file=sys.stderr)
        for name in failed:
            print(f"  - {name}", file=sys.stderr)


if __name__ == "__main__":
    main()
