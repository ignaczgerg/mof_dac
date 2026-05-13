from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from collections.abc import Iterable, Mapping

import pandas as pd
from fetch_wandb import fetch_wandb_summary

FLAG_TOKENS = {"scratch", "direct", "adabin", "noPBC"}
SIZE_TOKENS = {"sml", "med", "lrg", "vsml"}
BASIS_TOKENS = {
    "gaussian", "coulomb_sturmian", "cs", "bessel", 
    "laplace", "hankel", "radial_mlp", "cheby"
}

DEFAULT_DATASETS: Mapping[str, list[str]] = {
    "QM9":  ["U0", "G", "H", "U", "ZPVE", "cv", "mu", "qm9"],
    "RMD17": ['aspirin', 'benzene', 'ethanol', 'paracetamol',
              'salicylic', 'uracil', 'naphthalene'],
    "MD22": ['Ac-Ala3-NHMe'],
    "DB1": ["adsorption_db_1"],
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate latex performance tables from wandb summaries.")
        
    p.add_argument("--entity", default="theme4", help="wandb entity")
    p.add_argument("--project", default="aramco_dac", help="wandb project")
    p.add_argument("--datasets", nargs="+",
                   choices=list(DEFAULT_DATASETS.keys()),
                   help="Datasets to include; default = all")
    p.add_argument("--targets", nargs="+",
                   help="Override target list")
    p.add_argument("--no-train", dest="train", action="store_false",
                   help="Exclude train metrics")
    p.add_argument("--no-test", dest="test", action="store_false",
                   help="Exclude test metrics")
    p.add_argument("--no-val", dest="val", action="store_false",
                   help="Exclude val metrics")
    p.add_argument("-o", "--output", type=Path,
                   help="Write latex to file instead of console")
    p.set_defaults(train=False, test=True, val=True)
    return p.parse_args(argv)


def parse_run_name(name: str) -> dict:
    out = {}

    m = re.search(r'_PT[\[\(]([^\]\)]+)[\]\)]', name)
    if m:
        full = m.group(1)
        out["pretrain_full"] = full
        out["pretrain_source"] = full.split("_")[0]
    elif "_PT_" in name or name.endswith("_PT"):
        out["pretrain_source"] = None

    m = re.search(r'_ep(\d+)', name)
    if m:
        out["epochs"] = int(m.group(1))

    m = re.search(r'_LR([0-9eE\.\-]+)', name)
    if m:
        try:
            out["lr"] = float(m.group(1))
        except ValueError:
            out["lr_raw"] = m.group(1)

    m = re.search(r'_bs(\d+)', name)
    if m:
        out["batch_size"] = int(m.group(1))

    m = re.search(r'_(EqV\d+|gemnet[^_\[\]\(\)]*)', name)
    if m:
        out["architecture"] = m.group(1)

    m = re.search(r'_(sml|med|lrg|vsml|EqV2-\d+L)(?:_|$)', name)
    if m:
        out["size"] = m.group(1)

    m = re.search(r'_num-dist-base-(\d+)', name)
    if m:
        out["num_dist_base"] = int(m.group(1))

    basis_pattern = r'_(' + '|'.join(BASIS_TOKENS) + r')(?:_|$)'
    m = re.search(basis_pattern, name)
    if m:
        out["basis"] = m.group(1)

    for flag in FLAG_TOKENS:
        if f"_{flag}" in name:
            out[flag] = True

    for key, val in re.findall(r'([A-Za-z0-9]+)\[([^\]]+)\]', name):
        if key != "PT":
            out[key.lower()] = val

    for m in re.finditer(r'_(radius|largeN|fixedN|temp)(\d+(?:\.\d+)?)', name):
        out[m.group(1).lower()] = float(m.group(2))

    return out


def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Some magic cleaning to make the data readable."""
    data = data.copy()
    data['name'] = data['name'].str.replace('qm9_U_0_', 'qm9_U0_')
    data['name'] = data['name'].str.replace('qm9_c_v_', 'qm9_cv_')
    data['dataset'] = [i.split('_')[0] if '_' in i else i for i in data['name']]
    data['target'] = [i.split('_')[1] if '_' in i and len(i.split('_')) > 1 else 'unknown' for i in data['name']]
    
    data = data.drop([col for col in data.columns if col.startswith('cfg_')], axis=1)
    return data


def parse_by_targets(data: pd.DataFrame, /) -> dict[str, pd.DataFrame]:
    cleaned = _clean_data(data)
    parsed_meta = pd.json_normalize(cleaned["name"].apply(parse_run_name))
    df = pd.concat([cleaned, parsed_meta], axis=1)

    new_cols = {}
    for col in df.columns:
        if col.startswith('sum_'):
            _set = col.split('/')[0].replace('sum_', '')
            _metric = col.split('/')[-1]
            new_cols[col] = f"{_set}/{_metric}" if _metric else _set
        elif col.startswith('cfg_'):
            new_cols[col] = col.replace('cfg_', 'config_')
        else:
            new_cols[col] = col
    df.rename(columns=new_cols, inplace=True)

    keep_report = [
        "run_id", "dataset", "target", "architecture", "epochs",
        "pretrain_full", "scratch", "name", "created_at",
        "size", "num_dist_base", "basis"
    ]
    df = df[[c for c in df.columns if (
        ('train' in c or 'test' in c or 'val' in c) or (c in keep_report)
    )]].dropna(axis=1, how='all')

    out: dict[str, pd.DataFrame] = {}
    if 'target' in df.columns:
        for tgt, grp in df.groupby('target', sort=False):
            grp = grp.dropna(axis=1, how='all')
            grp = grp.loc[:, ~grp.columns.duplicated()]
            out[tgt] = grp.reset_index(drop=True)
    return out


def get_report(d: Mapping[str, pd.DataFrame],
               target: str,
               /,
               *,
               short: bool = True,
               sort: bool = True,
               sort_by: list[str] | None = None,
               train: bool = False,
               test: bool = True,
               val: bool = False) -> pd.DataFrame:
    if target not in d:
        return pd.DataFrame()
        
    df = d[target]

    cols: list[str] = []
    if short:
        cols += ["run_id", "dataset", "architecture", "size", 
                 "num_dist_base", "basis", "pretrain_full"]
    if train:
        cols += [c for c in df.columns if c.startswith("train")]
    if test:
        cols += [c for c in df.columns if c.startswith("test")]
    if val:
        cols += [c for c in df.columns if c.startswith("val")]

    df = df[[c for c in cols if c in df.columns]]

    if sort and not df.empty:
        try:
            if sort_by:
                df = df.sort_values(by=sort_by, ascending=True)
            else:
                metric_candidates = [c for c in df.columns
                                     if c.startswith(("test/", "val/", "train/"))]
                if metric_candidates:
                    df = df.sort_values(by=metric_candidates[0], ascending=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not sort ({exc}). Returning unsorted df.", file=sys.stderr)
    return df


def sanitize_latex(text: str | object) -> str | object:
    if isinstance(text, (int, float)):
        return text
    text = str(text)
    return text.replace('_', r'\_')


def render_latex_tables(results: Mapping[str, pd.DataFrame],
                        *,
                        datasets: Mapping[str, list[str]],
                        train: bool,
                        test: bool,
                        val: bool,
                        file=sys.stdout) -> None:
    for ds_name, targets in datasets.items():
        print(f"\\section{{Dataset: {sanitize_latex(ds_name)}}}", file=file)
        for target in targets:
            print(f"\\subsection{{Target: {sanitize_latex(target)}}}", file=file)

            report = get_report(results, target,
                                short=True, train=train, test=test, val=val)
            
            if report.empty:
                print("No data found.", file=file)
                continue

            report = report.rename(columns=sanitize_latex)
            report = report.map(sanitize_latex)
            report = report.fillna("-")

            fmt = 'l' + 'c' * (len(report.columns) - 1 or 0)
            print(report.to_latex(index=False, escape=False, column_format=fmt),
                  file=file)
            print("\n\n", file=file)


def fetch_results(
        entity: str = "theme4",
        project: str = "aramco_dac",
        *,
        use_history: bool = False) -> dict[str, pd.DataFrame]:
    df_raw = fetch_wandb_summary(entity=entity,
                                project=project,
                                use_history=use_history)
    return parse_by_targets(df_raw)


def pretty_tables(results: dict[str, pd.DataFrame],
                  *,
                  datasets: Mapping[str, list[str]] | None = None,
                  train: bool = False,
                  test: bool = True,
                  val: bool = False) -> str:

    from io import StringIO

    buf = StringIO()
    render_latex_tables(results,
                        datasets=datasets or DEFAULT_DATASETS,
                        train=train, test=test, val=val,
                        file=buf)
    return buf.getvalue()


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    results = fetch_results(entity=args.entity,
                            project=args.project)

    datasets = {k: (args.targets or v)
                for k, v in DEFAULT_DATASETS.items()
                if args.datasets is None or k in args.datasets}

    latex_blob = pretty_tables(results,
                               datasets=datasets,
                               train=args.train,
                               test=args.test,
                               val=args.val)

    if args.output:
        args.output.write_text(latex_blob)
        print(f"latex written to {args.output}", file=sys.stderr)
    else:
        print(latex_blob)


if __name__ == "__main__":
    main()