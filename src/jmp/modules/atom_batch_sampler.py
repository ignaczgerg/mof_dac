from __future__ import annotations
import random
from typing import List, Optional, Sequence
from torch.utils.data import Sampler, BatchSampler
import torch
import torch.distributed as dist

class AtomBucketBatchSampler(BatchSampler):
    """
    Packs indices so that sum(sizes[idx]) <= max_atoms_per_batch.
    Bucketed, per-epoch shuffling. DDP-safe (equalizes steps across ranks).
    """

    def __init__(
        self,
        base_sampler: Sampler[int],
        sizes: Sequence[int],
        max_atoms_per_batch: int,
        bucket_boundaries: Optional[Sequence[int]] = (128, 256, 384, 512, 768, 1024, 1536, 2048),
        drop_last: bool = True,
        seed: int = 0,
    ):
        if max_atoms_per_batch <= 0:
            raise ValueError("max_atoms_per_batch must be > 0")
        self.base_sampler = base_sampler
        self.sampler = base_sampler
        self.sizes = sizes
        self.cap = int(max_atoms_per_batch)
        self.boundaries = list(bucket_boundaries) if bucket_boundaries else []
        self.drop_last = drop_last
        self.seed = seed

        self._batches: List[List[int]] = []
        self._epoch = 0
        self._built_for_epoch: Optional[int] = None

    def _bucket_id(self, n: int) -> int:
        for i, b in enumerate(self.boundaries):
            if n <= b:
                return i
        return len(self.boundaries)

    def _epoch_for_seed(self) -> int:
        e = getattr(self.base_sampler, "epoch", None)
        return int(e) if isinstance(e, int) else self._epoch

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._batches = []
        self._built_for_epoch = None
        if hasattr(self.base_sampler, "set_epoch"):
            self.base_sampler.set_epoch(epoch)

    def _build_batches_for_epoch(self, epoch_seed: int):
        indices = list(iter(self.base_sampler))
        rng = random.Random(self.seed + 17 * epoch_seed)

        if self.boundaries:
            buckets: List[List[int]] = [[] for _ in range(len(self.boundaries) + 1)]
            for i in indices:
                buckets[self._bucket_id(int(self.sizes[i]))].append(i)
            for b in buckets:
                rng.shuffle(b)
            stream = [i for b in buckets for i in b]
        else:
            stream = indices
            # no need to shuffle again, 
            # we shuffled the stream before
            # rng.shuffle(stream)

        batches: List[List[int]] = []
        cur: List[int] = []
        cur_atoms = 0
        for i in stream:
            s = int(self.sizes[i])
            if s > self.cap:
                if cur:
                    batches.append(cur)
                batches.append([i])
                cur, cur_atoms = [], 0
                continue

            if cur_atoms + s <= self.cap:
                cur.append(i)
                cur_atoms += s
            else:
                if cur:
                    batches.append(cur)
                cur, cur_atoms = [i], s

        if cur and not self.drop_last:
            batches.append(cur)

        rng.shuffle(batches)
        self._batches = batches
        self._built_for_epoch = epoch_seed

        # Equalize num steps across ranks (truncate to min)
        if dist.is_available() and dist.is_initialized():
            backend = str(dist.get_backend()).lower()
            dev = torch.device("cuda") if backend == "nccl" and torch.cuda.is_available() else torch.device("cpu")
            local = torch.tensor([len(self._batches)], device=dev)
            gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local)
            min_len = int(torch.stack(gathered).min().item())
            self._batches = self._batches[:min_len]

    def __iter__(self):
        if not hasattr(self.base_sampler, "epoch"):
            self._epoch += 1

        e = self._epoch_for_seed()
        if self._built_for_epoch != e:
            self._build_batches_for_epoch(e)

        for batch in self._batches:
            yield batch

    def __len__(self) -> int:
        e = self._epoch_for_seed()
        if not self._batches or self._built_for_epoch != e:
            self._build_batches_for_epoch(e)
        return len(self._batches)
