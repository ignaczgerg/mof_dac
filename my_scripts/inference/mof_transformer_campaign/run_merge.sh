#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name=moftr-merge
#SBATCH -o output/merge.%j.out
#SBATCH -e output/merge.%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

conda activate aramco_dac
cd "$SLURM_SUBMIT_DIR"

: "${CAMPAIGN_DIR:=/ibex/project/c2261/dac_iclr/predictions/mof_transformer_ablation_F2}"
: "${NUM_SHARDS:=150}"

DONE_N=$(ls -d "$CAMPAIGN_DIR"/shards/shard_*_of_$(printf '%04d' "$NUM_SHARDS")/DONE 2>/dev/null | wc -l)
echo "shards DONE: $DONE_N/$NUM_SHARDS"
if [ "$DONE_N" -ne "$NUM_SHARDS" ]; then
    echo "refusing to merge: incomplete campaign"
    exit 3
fi

rm -rf "$CAMPAIGN_DIR/merged"
python "$SLURM_SUBMIT_DIR/merge_shards.py" "$CAMPAIGN_DIR/shards" --merged-subdir ../merged || exit 4

python - "$CAMPAIGN_DIR" "$NUM_SHARDS" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]); n = int(sys.argv[2])
metas = [json.loads(p.read_text()) for p in sorted((root / "shards").glob(f"shard_*_of_{n:04d}/shard_meta.json"))]
kept = sum(m["kept"] for m in metas); dropped = sum(m["dropped"] for m in metas); files = sum(m["files_in_shard"] for m in metas)
ref_sid = None; report = {"shards": len(metas), "files": files, "kept": kept, "dropped": dropped, "targets": {}}
for f in sorted((root / "merged").glob("*.npz")):
    d = np.load(f, allow_pickle=True); sid = d["sid"]; pred = d["pred"]
    if ref_sid is None:
        ref_sid = sid
    ok = sid.shape[0] == kept and len(set(sid.tolist())) == kept and np.array_equal(sid, ref_sid) and np.isfinite(pred).all()
    report["targets"][f.stem] = {"rows": int(sid.shape[0]), "nonfinite": int((~np.isfinite(pred)).sum()), "ok": bool(ok)}
    if not ok:
        print(json.dumps(report, indent=1)); sys.exit(f"verification failed for {f.name}")
(root / "merged" / "verification.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
PY
