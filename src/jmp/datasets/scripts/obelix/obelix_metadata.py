import lmdb, pickle, numpy as np, os
from torch_geometric.data import Data

def collect_natoms(lmdb_path):
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    natoms = []
    with env.begin() as txn:
        for k, v in txn.cursor():
            if k == b"length":  # skip meta
                continue
            d: Data = pickle.loads(v)
            n = getattr(d, "natoms", getattr(d, "num_nodes", None))
            if n is None:
                n = d.atomic_numbers.shape[0]
            natoms.append(int(n))
    env.close()
    return np.array(natoms)

for split in ["train","val","test"]:
    p = f"/ibex/project/c2261/datasets/obelix/lmdb/{split}"
    lmdb_files = [f for f in os.listdir(p) if f.endswith(".lmdb")]
    assert len(lmdb_files) == 1
    natoms = collect_natoms(os.path.join(p, lmdb_files[0]))
    np.savez(os.path.join(p, "metadata.npz"), natoms=natoms)
