# SOAP Descriptor Pipeline

This folder contains utilities to compute SOAP descriptors from CIF files and export them to CSV.

## Main script

- `process_cifs.py`: Processes a directory of `.cif` files and writes SOAP descriptors to a CSV file.

## Quick start

Run from the project root:

```bash
python soap/process_cifs.py \
  --input-dir soap/test_cifs \
  --output-csv soap/output_descriptors.csv \
  --recursive
```

## Common options

- `--input-dir`: Directory containing CIF files.
- `--output-csv`: Path to output CSV.
- `--recursive`: Recursively search subfolders for CIF files.
- `--n-jobs`: Number of parallel workers (default: `-1`).
- `--timelimit`: Per-file timeout in seconds (default: `240`).
- `--no-obabel-preconversion`: Disable Open Babel pre-conversion.
- `--json-config`: Path to SOAP config JSON (default: `soap/soap_config.json`).
- `--properties-json`: Path to species properties JSON (default: `soap/species_default_properties.json`).

## Notes

- Python dependencies include `dscribe`, `pymatgen`, `pandas`, `joblib`, and `mendeleev`.
- Open Babel (`obabel`) is required unless `--no-obabel-preconversion` is used.
