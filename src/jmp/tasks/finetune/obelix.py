import os
import numpy as np
from typing import Annotated, Literal, TypeAlias, final, Union
from typing_extensions import override, TypeVar
from typing_extensions import override
import torch
import copy
from torch_geometric.data.data import BaseData
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, get_worker_info
from jmp.lightning import Field, TypedConfig
from jmp.tasks.finetune.model_wrapper_base import (
    ModelWrapperBase,
    EScAIPModelWrapper,
)
from jmp.modules.transforms.normalize import (
    NormalizationConfig,
    PositionNormalizationConfig,
)
from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from .base import FinetuneConfigBase, FinetuneModelBase

from functools import partial
from torch_geometric.data.batch import Batch
from jmp.modules.atom_batch_sampler import AtomBucketBatchSampler

import jmp.tasks.finetune.obelix as obelix
import sys, inspect

NormalizationUnion = Union[NormalizationConfig, PositionNormalizationConfig]

ObelixTarget: TypeAlias = Literal[
    "ionic_conductivity",
]

class DefaultOutputHeadConfig(TypedConfig):
    name: Literal["default"] = "default"

OutputHeadConfig = Annotated[DefaultOutputHeadConfig, Field(discriminator="name")]

class ObelixConfig(FinetuneConfigBase):
    """Finetune config for Obelix graph-level regression target."""

    normalization: dict[str, NormalizationUnion] = Field(default_factory=dict, discriminator="type")

    graph_scalar_targets: list[ObelixTarget] = []
    node_vector_targets: list[str] = []

    graph_scalar_reduction: dict[str, Literal["sum", "mean", "max"]] = {
        "ionic_conductivity": "mean",
    }
    graph_scalar_reduction_default: Literal["sum", "mean", "max"] = "mean"

    output_head: OutputHeadConfig = DefaultOutputHeadConfig()
    max_neighbors: int = 30


TModel = TypeVar("TModel", bound=ModelWrapperBase, infer_variance=True)

def _seed_worker(worker_id: int):
    base_seed = torch.initial_seed() % 2**32
    try:
        import random
        np.random.seed(base_seed + worker_id)
        random.seed(base_seed + worker_id)
    except Exception:
        pass

def _collate_exclude(data_list, *, exclude_keys):
    return Batch.from_data_list(data_list, exclude_keys=exclude_keys)

@final
class ObelixModel(FinetuneModelBase[ObelixConfig, TModel]):
    """Model wrapper for Obelix finetuning."""

    targets: list[ObelixTarget] = [
        "ionic_conductivity",
    ]

    def _unwrap_base(self, ds):
        seen = set()
        while hasattr(ds, "dataset") and id(ds) not in seen:
            seen.add(id(ds))
            ds = ds.dataset
        return ds

    def _collect_natoms_list(self, ds) -> list[int]:
        for attr in ("natoms", "natoms_per_item", "sizes"):
            if hasattr(ds, attr):
                try:
                    arr = getattr(ds, attr)
                    return [int(x) for x in list(arr)]
                except Exception:
                    pass
        out = []
        for i in range(len(ds)):
            item = ds[i]
            out.append(int(getattr(item, "natoms", getattr(item, "num_nodes", item.atomic_numbers.shape[0]))))
        return out

    @classmethod
    @override
    def config_cls(cls):
        return ObelixConfig

    @override
    def metric_prefix(self) -> str:
        return "obelix"

    @override
    def validate_config(self, config: ObelixConfig):
        super().validate_config(config)
        for key in config.graph_scalar_targets:
            assert key in self.targets, f"{key} is not a valid Obelix target"

    @override
    def construct_graph_scalar_output_head(self, target: str):
        return super().construct_graph_scalar_output_head(target)

    @override
    def process_aint_graph(self, aint_graph: Graph):
        return aint_graph

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)


        if not hasattr(data, "natoms"):
            data.natoms = getattr(data, "num_nodes", data.atomic_numbers.shape[0])
        if not hasattr(data, "tags"):
            data.tags = torch.full((data.natoms,), 2, dtype=torch.long)

        if hasattr(data, "cell") and isinstance(data.cell, torch.Tensor) and data.cell.ndim == 2:
            data.cell = data.cell.unsqueeze(0)

        if self.config.model_cls == EScAIPModelWrapper:
            return data

        cutoff = self.config.backbone.max_radius
        if "pos" in self.config.normalization:
            pos_std = self.config.normalization["pos"]["std"]
            cutoff = cutoff / pos_std

        pbc = bool(getattr(self.config, "meta", {}).get("pbc_default", True))

        data = self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(cutoff),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(self.config.backbone.max_neighbors),
            pbc=pbc,
        )
        return data
    
    def _natoms_per_item(self, dataset) -> list[int]:
        base = self._unwrap_base(dataset)
        return self._collect_natoms_list(base)

    def _make_loader(self, dataset, *, train: bool):
        sizes = self._natoms_per_item(dataset)
        assert len(sizes) == len(dataset), "sizes/dataset length mismatch"

        args = getattr(self.config, "args", None)
        num_workers = int(getattr(args, "num_workers", 4)) if args is not None else 4
        timeout_s = int(getattr(args, "dataloader_timeout_s", 120)) if args is not None else 120
        seed = int(getattr(args, "seed", 0)) if args is not None else 0
        pin_memory = bool(getattr(self.config, "pin_memory", True))
        exclude_keys = getattr(self.config, "exclude_keys", []) or []

        cap = getattr(args, "max_atoms_per_batch", None) if args is not None else None
        if cap is None:
            cap = int(os.getenv("MAX_ATOMS_PER_BATCH", "0")) or int(np.median(sizes) * max(1, self.config.batch_size))
        cap = int(cap)

        world_size = int(getattr(self.trainer, "world_size", 1))
        if world_size > 1:
            base_sampler = DistributedSampler(
                dataset,
                shuffle=train,
                drop_last=True if train else False,
            )
        else:
            base_sampler = RandomSampler(dataset)

        batch_sampler = AtomBucketBatchSampler(
            base_sampler=base_sampler,
            sizes=sizes,
            max_atoms_per_batch=cap,
            bucket_boundaries=(128, 256, 384, 512, 768, 1024, 1536, 2048),
            drop_last=True if train else False,
            seed=seed,
        )

        if hasattr(batch_sampler, "__len__") and len(batch_sampler) == 0:
            raise RuntimeError(
                f"AtomBucketBatchSampler built 0 batches (cap={cap}). "
                f"Increase cap or set drop_last=False for debugging."
            )

        dl_kwargs = dict(
            dataset=dataset,
            batch_sampler=batch_sampler,
            pin_memory=pin_memory,
            collate_fn=partial(_collate_exclude, exclude_keys=exclude_keys),
        )

        if num_workers > 0:
            dl_kwargs.update(
                num_workers=num_workers,
                persistent_workers=True,
                worker_init_fn=_seed_worker,
                timeout=timeout_s,
                prefetch_factor=4,
            )
        else:
            dl_kwargs.update(num_workers=0, timeout=0)

        return DataLoader(**dl_kwargs)
    
    def train_dataloader(self):
        args = getattr(self.config, "args", None)
        use_bucket = bool(getattr(args, "atom_bucket_batch_sampler", False)) \
                     or bool(getattr(self.config, "use_atom_bucket_batch_sampler", False))
        if not use_bucket:
            return super().train_dataloader()
        ds = self.train_dataset()
        return self._make_loader(ds, train=True)

    def val_dataloader(self):
        args = getattr(self.config, "args", None)
        use_bucket = bool(getattr(args, "atom_bucket_batch_sampler", False)) \
                     or bool(getattr(self.config, "use_atom_bucket_batch_sampler", False))
        if not use_bucket:
            return super().val_dataloader()
        ds = self.val_dataset()
        return self._make_loader(ds, train=False)

    def test_dataloader(self):
        args = getattr(self.config, "args", None)
        use_bucket = bool(getattr(args, "atom_bucket_batch_sampler", False)) \
                     or bool(getattr(self.config, "use_atom_bucket_batch_sampler", False))
        if not use_bucket:
            return super().test_dataloader()
        ds = self.test_dataset()
        return self._make_loader(ds, train=False)