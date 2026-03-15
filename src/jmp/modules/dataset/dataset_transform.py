"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from collections import abc
from collections.abc import Callable
from functools import cache, partial
from logging import config, getLogger
from typing import Any, cast, Optional, Protocol, Callable
 
import numpy as np
import torch
import wrapt
from typing_extensions import override

from .. import transforms as T
from .dataset_typing import TDataset
from tqdm import tqdm
from mpire import WorkerPool

log = getLogger(__name__)


def transform(
    dataset: TDataset,
    transform: Callable[[Any], Any],
    copy_data: bool = True,
) -> TDataset:
    """
    Applies a transformation/mapping function to all elements of the dataset.

    Args:
        dataset (Dataset): The dataset to transform.
        transform (Callable): The transformation function.
        copy_data (bool, optional): Whether to copy the data before transforming. Defaults to True.
    """

    class _TransformedDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx):
            nonlocal copy_data, transform

            assert transform is not None, "Transform must be defined."
            data = self.__wrapped__.__getitem__(idx)
            if copy_data:
                data = copy.deepcopy(data)
            data = transform(data)
            return data

    return cast(TDataset, _TransformedDataset(dataset))


def atomref_transform(
    dataset: TDataset,
    refs: dict[str, torch.Tensor],
    keep_raw: bool = False,
) -> TDataset:
    """
    Subtracts the atomrefs from the target properties of the dataset. For a data sample x and atomref property p,
    the transformed property is `x[p] = x[p] - atomref[x.atomic_numbers].sum()`.

    This is primarily used to normalize energies using a "linear referencing" scheme.

    Args:
        dataset (Dataset): The dataset to transform.
        refs (dict[str, torch.Tensor]): The atomrefs to subtract from the target properties.
        keep_raw (bool, optional): Whether to keep the original properties, renamed as `{target}_raw`. Defaults to False.
    """
    # Convert the refs to tensors
    refs_dict: dict[str, torch.Tensor] = {}
    for k, v in refs.items():
        if isinstance(v, list):
            v = torch.tensor(v)
        elif isinstance(v, np.ndarray):
            v = torch.from_numpy(v).float()
        elif not torch.is_tensor(v):
            raise TypeError(f"Invalid type for {k} in atomrefs: {type(v)}")
        refs_dict[k] = v

    return transform(
        dataset,
        partial(T.atomref_transform, refs=refs_dict, keep_raw=keep_raw),
        copy_data=False,
    )


def expand_dataset(dataset: TDataset, n: int) -> TDataset:
    """
    Expands the dataset to have `n` elements by repeating the elements of the dataset as many times as necessary.

    Args:
        dataset (Dataset): The dataset to expand.
        n (int): The desired length of the dataset.
    """
    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"expand_dataset ({n}) must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    og_size = len(dataset)
    if og_size > n:
        raise ValueError(
            f"expand_dataset ({n}) must be greater than or equal to the length of the dataset "
            f"({len(dataset)}) for {dataset.__class__.__qualname__}"
        )

    class _ExpandedDataset(wrapt.ObjectProxy):
        @override
        def __len__(self):
            nonlocal n
            return n

        @override
        def __getitem__(self, index: int):
            nonlocal n, og_size
            if index < 0 or index >= n:
                raise IndexError(
                    f"Index {index} is out of bounds for dataset of size {n}."
                )
            return self.__wrapped__.__getitem__(index % og_size)

        @cache
        def _atoms_metadata_cached(self):
            """
            We want to retrieve the atoms metadata for the expanded dataset.
            This includes repeating the atoms metadata for the elemens that are repeated.
            """

            # the out metadata shape should be (n,)
            nonlocal n, og_size

            metadata = self.__wrapped__.atoms_metadata
            metadata = np.resize(metadata, (n,))
            log.debug(
                f"Expanded the atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    dataset = cast(TDataset, _ExpandedDataset(dataset))
    log.info(f"Expanded dataset {dataset.__class__.__name__} from {og_size} to {n}.")
    return dataset


def first_n_transform(dataset: TDataset, *, n: int) -> TDataset:
    """
    Returns a new dataset that contains the first `n` elements of the original dataset.

    Args:
        dataset (Dataset): The dataset to transform.
        n (int): The number of elements to keep.
    """
    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"first_n ({n}) must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    if len(dataset) < n:
        raise ValueError(
            f"first_n ({n}) must be less than or equal to the length of the dataset "
            f"({len(dataset)}) for {dataset.__class__.__qualname__} "
        )

    class _FirstNDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx: int):
            nonlocal n

            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Index {idx} is out of bounds for dataset of size {n}."
                )

            return self.__wrapped__.__getitem__(idx)

        @override
        def __len__(self):
            nonlocal n
            return n

        @cache
        def _atoms_metadata_cached(self):
            """We only want to retrieve the atoms metadata for the first n elements."""
            nonlocal n

            metadata = self.__wrapped__.atoms_metadata
            og_size = len(metadata)
            metadata = metadata[:n]

            log.info(
                f"Retrieved the first {n} atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    return cast(TDataset, _FirstNDataset(dataset))


def sample_n_transform(dataset: TDataset, *, n: int, seed: int, max_natoms: int | None = None) -> TDataset:
    """
    Similar to first_n_transform, but samples n elements randomly from the dataset.

    Args:
        dataset (Dataset): The dataset to transform.
        n (int): The number of elements to sample.
        seed (int): The random seed to use for sampling.
        max_natoms (int, optional): If provided, filters the dataset to only include samples
            with a number of atoms less than this value before sampling.
    """

    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"sample_n ({n}) must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )
    
    if max_natoms is not None:
        if not hasattr(dataset, "atoms_metadata"):
            raise AttributeError(
                f"Dataset of type {type(dataset)} does not have 'atoms_metadata', required for max_natoms filtering."
            )
        valid_indices = np.where(dataset.atoms_metadata < max_natoms)[0]
    else:
        valid_indices = np.arange(len(dataset))

    if len(valid_indices) < n:
        raise ValueError(
            f"sample_n ({n}) must be less than or equal to the length of the dataset "
            f"({len(dataset)}) for {dataset.__class__.__qualname__} "
        )
    sampled_indices = np.random.default_rng(seed).choice(valid_indices, n, replace=False)

    class _SampleNDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx: int):
            nonlocal n, sampled_indices

            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Index {idx} is out of bounds for dataset of size {n}."
                )

            return self.__wrapped__.__getitem__(sampled_indices[idx])

        @override
        def __len__(self):
            nonlocal n
            return n

        @cache
        def _atoms_metadata_cached(self):
            """We only want to retrieve the atoms metadata for the sampled n elements."""
            nonlocal n, sampled_indices

            metadata = self.__wrapped__.atoms_metadata
            og_size = len(metadata)
            metadata = metadata[sampled_indices]

            log.info(
                f"Retrieved the sampled {n} atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    return cast(TDataset, _SampleNDataset(dataset))


def filter_transform(dataset: TDataset, *, predicate: Callable[[Any], bool], 
                     filtered_indices: list[int] | None = None) -> TDataset:
    """
    Transforms the dataset by filtering out elements that do not satisfy the given predicate.
    
    Args:
        dataset (Dataset): The dataset to transform.
        predicate (Callable[[Any], bool]): A function that takes an element (or its metadata)
                                             and returns True if the element should be kept.
        filtered_indices (list[int] | None): A list of indices that satisfy the predicate.
                                              If None, the indices are computed from the predicate.
    """
    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"filter_transform must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__}"
        )
    
    if filtered_indices is None:
        filtered_indices = [i for i in range(len(dataset)) if predicate(dataset[i])]
    
    if (n := len(filtered_indices)) == 0:
        raise ValueError("filter_transform: No items match the filter condition.")
     
    class _FilterDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx: int):
            nonlocal n, filtered_indices
            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Index {idx} is out of bounds for filtered dataset of size {n}."
                )
            return self.__wrapped__.__getitem__(filtered_indices[idx])
    
        @override
        def __len__(self):
            return n
    
    
    return cast(TDataset, _FilterDataset(dataset))


def compute_quantile_bins(y: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    Return n_bins+1 edges at the empirical quantiles of y.
    """
    
    q_list = torch.linspace(0.0, 1.0, steps=n_bins + 1, 
                    device=y.device, dtype=y.dtype)
    
    return torch.quantile(y, q_list)

def discretize_target(y: torch.Tensor, bin_edges: torch.Tensor) -> torch.LongTensor:
    """
    Map each scalar y -> its bin index in [0..n_bins-1].
    """
    idx = torch.bucketize(y, bin_edges) - 1
    # idx = torch.tensor(idx.clamp(0, bin_edges.size(0) - 2).tolist(), device=idx.device, dtype=torch.long)
    idx = idx.clamp(0, bin_edges.size(0) - 2).long()
    return idx


def adaptive_binning_transform(
    dataset: TDataset,
    targets: list[str],
    n_bins: list[int],
    keep_raw: bool = True,
    copy_data: bool = False,
    bin_edges: list[torch.Tensor] | None = None,
) -> tuple[TDataset, list[torch.Tensor]]:
    """
    Transforms a regression target dataset into a classification dataset by binning the target values.
    
    Args:
        dataset: the dataset to transform.
        target: the name of the target property to bin.
        n_bins: number of classes (bins).
        keep_raw: if True, stores the original `target` as `data.target_raw`.
            This is added so that regression metrics can be computed on the original target.
        copy_data: whether to deepcopy each sample before modifying.

    Returns:
        transformed dataset and bin_edges [to compute regression metrics]
    """
    if not isinstance(dataset, abc.Sized):
        raise TypeError("adaptive_binning_transform requires a dataset with __len__")
    
    if bin_edges is None:
        bin_edges = []
        for target, _n_bins in zip(targets, n_bins):
    
            ys = []
            for i in range(len(dataset)):
                sample = dataset[i]
                y = getattr(sample, target) 
                ys.append(torch.as_tensor(y))

            y_all = torch.stack(ys)
            bin_edges.append(compute_quantile_bins(y_all, _n_bins))

    class _BinnedDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx: int) -> Any:
            nonlocal copy_data, bin_edges, keep_raw, targets
            assert isinstance(bin_edges[0], torch.Tensor), "bin_edges must be a tensor"
            data = self.__wrapped__[idx]
            if copy_data:
                data = copy.deepcopy(data)
            
            # original y → bin index
            for bin_edges_idx, target in enumerate(targets):
                raw_y = getattr(data, target) 
                y_tensor = torch.as_tensor(raw_y)
                bin_idx = discretize_target(y_tensor, bin_edges[bin_edges_idx])
                
                if keep_raw:
                    setattr(data, f"{target}_raw", raw_y)

                # overwrite
                setattr(data, target, bin_idx)

            return data
        
        @override
        def __len__(self):
            return len(self.__wrapped__)
    
    return _BinnedDataset(dataset), bin_edges


def set_normalization_config(
    dataset: TDataset,
    model: Any, 
    targets: list[str],
):
    """
    Configures the mean and std of the scalar (regression) targets in the dataset.
    """
    
    from jmp.modules.transforms.normalize import NormalizationConfig as NC
    
    # TODO: optimize this if the dataset becomes large enough
    samples_dict = {target: [] for target in targets}
    for i in range(len(dataset)):
        sample = dataset[i]
        for target in targets:
            samples_dict[target].append(getattr(sample, target))
    
    # TODO: support different normalization types for different targets
    model.config.normalization = {
        target: NC(
            mean=float(np.mean(samples_dict[target])),
            std=float(np.std(samples_dict[target])),
        )
        for target in targets
    }
    
    # print("Updated normalization configuration after filtering:", 
    #       model.config.normalization) 

    return 


def set_task_otf_normalization_config(
    dataset: TDataset,
    model: Any,
    targets: list[str],
    task: Any, # TaskConfig
    heteroscedastic: bool | None = None,
):
    """
    Configures the mean and std of targets in the dataset.

    If heteroscedastic mode is enabled (either via parameter or model.config.heteroscedastic),
    also computes normalization stats for {target}_std fields using log normalization.
    """
    from tqdm import tqdm
    from jmp.modules.transforms.normalize import NormalizationConfig as NC

    print("normalization configuration before filtering:",
        task.normalization)

    if heteroscedastic is None:
        heteroscedastic = getattr(getattr(model, 'config', None), 'heteroscedastic', False)

    _transform_if_log = lambda x, norm: np.log(np.abs(x) + 1e-12) if norm.normalization_type == "log" else x

    all_targets = list(targets)
    std_targets = []
    if heteroscedastic:
        for target in targets:
            std_target = f"{target}_std"
            std_targets.append(std_target)
            all_targets.append(std_target)

    samples_dict = {target: [] for target in all_targets}

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        for target in targets:
            samples_dict[target].append(
                _transform_if_log(getattr(sample, target), task.normalization.get(target, NC())))

        if heteroscedastic:
            for std_target in std_targets:
                if hasattr(sample, std_target):
                    val = getattr(sample, std_target)
                    samples_dict[std_target].append(np.log(np.abs(val) + 1e-12))

    task.normalization = {
        target: NC(
            mean=float(np.mean(samples_dict[target])),
            std=float(np.std(samples_dict[target])),
            normalization_type=task.normalization.get(target, NC()).normalization_type
        )
        for target in targets
    }

    if heteroscedastic:
        for std_target in std_targets:
            if samples_dict[std_target]:
                task.normalization[std_target] = NC(
                    mean=float(np.mean(samples_dict[std_target])),
                    std=float(np.std(samples_dict[std_target])),
                    normalization_type="log"
                )
                print(f"Computed OTF normalization for {std_target}: "
                      f"mean={task.normalization[std_target].mean:.6f}, "
                      f"std={task.normalization[std_target].std:.6f}")

    print("Updated normalization configuration after OTF computation:",
          task.normalization)

    return


def set_task_normalization_config(
    dataset: TDataset,
    config: Any, # TaskConfig
):
    """
    Configures the mean and std of the scalar (regression) targets in the dataset.
    """
    
    # from jmp.modules.transforms.normalize import NormalizationConfig as NC
    # from jmp.modules.transforms.normalize import PositionNormalizationConfig as NC

    samples_dict = {"pos": []}

    def process_sample(i):
        sample = dataset[i]
        sample_pos = sample.pos
        pos_centered = sample_pos - sample.center  # shape: (num_atoms, 3)
        return pos_centered.numpy()
    
    with WorkerPool(n_jobs=10) as pool:  # adjust number of workers based on CPU cores
        results = pool.map(process_sample, range(len(dataset)), progress_bar=True)
    
    samples_dict["pos"].extend([torch.from_numpy(r.reshape(-1, 3)) for r in results])
    
    # # TODO: optimize this if the dataset becomes large enough
    # samples_dict = {"pos": []}
    # for i in tqdm(range(len(dataset))):
    #     sample = dataset[i]
    #     sample_pos = sample.pos
    #     pos_centered = sample_pos - sample.center_pos             # shape: (num_atoms, 3)
    #     samples_dict["pos"].append(pos_centered)

    # TODO: support different normalization types for different targets
    # TODO: support more than one target
    # mean = np.mean(samples_dict_sum["pos"])
    all_pos = torch.cat(samples_dict["pos"], dim=0)
    # scalar_std = all_pos.norm(dim=1).std()  # shape: ()
    
    # config.normalization.update({
    #     "pos": NC(
    #         mean=0,
    #         std=float(scalar_std),
    #     )
    # })
    
    # print("Updated normalization configuration after filtering:", 
    #       config.normalization) 

    return all_pos




class _GraphLike(Protocol):
    edge_index: torch.Tensor
    pos: torch.Tensor

GraphBuilder = Callable[[Any], _GraphLike]

def _natoms_from_sample(x: Any) -> int:
    if hasattr(x, "natoms"):
        return int(x.natoms)
    if hasattr(x, "n_nodes"):
        return int(x.n_nodes)
    if hasattr(x, "pos"):
        return int(getattr(x, "pos").shape[0])
    raise AttributeError("Cannot infer number of atoms for a dataset item.")

def _natoms_list(dataset) -> list[int]:
    if hasattr(dataset, "atoms_metadata"):
        try:
            arr = getattr(dataset, "atoms_metadata")
            return [int(v) for v in np.asarray(arr).tolist()]
        except Exception:
            pass
    out = []
    for i in range(len(dataset)):
        out.append(_natoms_from_sample(dataset[i]))
    return out

@torch.no_grad()
def compute_dataset_stats(
    dataset,
    *,
    sample_size: int = 100,
    compute_degree: bool = True,
    graph_builder: Optional[GraphBuilder] = None,      # legacy path still supported
    pbar: bool = False,

    # NEW: call your mixin's generate_graph directly without helper
    generate_graph: Optional[Callable[..., Any]] = None,
    generate_graph_kwargs: Optional[dict] = None,
    ensure_batch: bool = True,
    require_cell_if_pbc: bool = True,
    force_otf: bool = True,
) -> dict[str, float]:
    """
    Returns:
      {
        'avg_num_nodes': float,
        'avg_degree': float,              # NaN if not computed
        'n_items': int,
        'n_degree_samples': int,
        'degree_fail_rate': float,
      }
    """
    sizes = _natoms_list(dataset)
    n_items = len(sizes)
    avg_num_nodes = float(np.mean(sizes)) if sizes else float("nan")

    avg_degree = float("nan")
    n_degree_samples = 0
    n_degree_fail = 0

    if compute_degree and n_items > 0 and sample_size > 0:
        step = max(1, n_items // min(sample_size, n_items))
        idxs = list(range(0, n_items, step))[:sample_size]
        it = tqdm(idxs, desc="Sampling degree") if pbar else idxs

        deg_vals: list[float] = []
        for i in it:
            try:
                x = dataset[i]

                # Build/obtain a graph-like object with .edge_index and .pos
                if graph_builder is not None:
                    g = graph_builder(x)

                elif generate_graph is not None:
                    s = x
                    # Sanitize: add batch for non-PBC path if missing
                    if ensure_batch and (not hasattr(s, "batch") or getattr(s, "batch") is None):
                        s.batch = torch.zeros(
                            int(s.pos.size(0)), dtype=torch.long, device=s.pos.device
                        )
                    # Ensure PBC preconditions when requested
                    kwargs = dict(generate_graph_kwargs or {})
                    if force_otf:
                        kwargs["otf_graph"] = True
                    if require_cell_if_pbc and bool(kwargs.get("use_pbc", False)) and not hasattr(s, "cell"):
                        n_degree_fail += 1
                        continue

                    g = generate_graph(s, **kwargs)  # expects GraphData with edge_index/pos

                else:
                    # Fallback: assume dataset item already *is* a graph
                    g = x

                if not hasattr(g, "edge_index") or not hasattr(g, "pos"):
                    n_degree_fail += 1
                    continue

                N = _natoms_from_sample(g)
                if N <= 0:
                    n_degree_fail += 1
                    continue

                src = g.edge_index[0]
                counts = torch.bincount(src.detach().cpu(), minlength=N)
                deg_vals.append(float(counts.sum().item() / max(1, N)))
                n_degree_samples += 1

            except Exception:
                n_degree_fail += 1
                continue

        if deg_vals:
            avg_degree = float(np.mean(deg_vals))

    stats = dict(
        avg_num_nodes=avg_num_nodes,
        avg_degree=avg_degree,
        n_items=n_items,
        n_degree_samples=n_degree_samples,
        degree_fail_rate=(n_degree_fail / max(1, (n_degree_samples + n_degree_fail))),
    )
    try:
        log.info(
            "[dataset_stats] N=%d  avg_num_nodes=%.3f  avg_degree=%s  samples=%d  fail_rate=%.2f",
            n_items,
            avg_num_nodes,
            f"{avg_degree:.3f}" if np.isfinite(avg_degree) else "NaN",
            n_degree_samples,
            stats["degree_fail_rate"],
        )
    except Exception:
        pass
    return stats

def with_dataset_stats(
    dataset,
    *,
    sample_size: int = 10000,
    compute_degree: bool = True,
    graph_builder: Optional[GraphBuilder] = None,

    # pass-through for direct generate_graph usage
    generate_graph: Optional[Callable[..., Any]] = None,
    generate_graph_kwargs: Optional[dict] = None,
    ensure_batch: bool = True,
    require_cell_if_pbc: bool = True,
    force_otf: bool = True,

    pbar: bool = False,
):
    """
    Wraps the dataset and attaches `.dataset_stats` and properties `.avg_num_nodes`, `.avg_degree`.
    """
    class _WithStats(wrapt.ObjectProxy):
        def __init__(self, wrapped):
            super().__init__(wrapped)
            s = compute_dataset_stats(
                wrapped,
                sample_size=sample_size,
                compute_degree=compute_degree,
                graph_builder=graph_builder,
                pbar=pbar,
                generate_graph=generate_graph,
                generate_graph_kwargs=generate_graph_kwargs,
                ensure_batch=ensure_batch,
                require_cell_if_pbc=require_cell_if_pbc,
                force_otf=force_otf,
            )
            object.__setattr__(self, "_stats", s)

        @property
        def dataset_stats(self) -> dict[str, float]:
            return getattr(self, "_stats")

        @property
        def avg_num_nodes(self) -> float:
            return float(self._stats.get("avg_num_nodes", float("nan")))

        @property
        def avg_degree(self) -> float:
            return float(self._stats.get("avg_degree", float("nan")))

    return cast(type(dataset), _WithStats(dataset))