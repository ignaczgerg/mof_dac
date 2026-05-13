"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from typing import Literal, final

import torch
from torch_geometric.data.data import BaseData
from typing_extensions import override, TypeVar

from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from .base import FinetuneConfigBase, FinetuneModelBase
from jmp.tasks.finetune.model_wrapper_base import ModelWrapperBase

from jmp.modules.transforms.normalize import NormalizationConfig, PositionNormalizationConfig
from typing import Union
from pydantic import Field

NormalizationUnion = Union[NormalizationConfig, PositionNormalizationConfig]


class QMOFConfig(FinetuneConfigBase):

    normalization: dict[str, NormalizationUnion] = Field(default_factory=dict, discriminator='type')
    graph_scalar_targets: list[str] = ["y"]
    node_vector_targets: list[str] = []

    graph_scalar_reduction_default: Literal["sum", "mean", "max"] = "mean"

TModel = TypeVar("TModel", bound=ModelWrapperBase, infer_variance=True)

@final
class QMOFModel(FinetuneModelBase[QMOFConfig, TModel]):
    @classmethod
    @override
    def config_cls(cls):
        return QMOFConfig

    @override
    def metric_prefix(self) -> str:
        return "qmof"

    @override
    def training_step(self, batch, batch_idx):
        with self.log_context(prefix=f"train/{self.metric_prefix()}/"):
            preds = self(batch)

            loss = self.compute_losses(batch, preds)
            self.log_dict(self.train_metrics(batch, preds))
            
            torch.cuda.empty_cache()

        return loss

    @override
    def process_aint_graph(self, aint_graph: Graph):
        return aint_graph

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)

        data = copy.deepcopy(data)
        if not torch.is_tensor(data.y):
            data.y = torch.tensor(data.y)
        data.y = data.y.view(-1)
        data.atomic_numbers = data.atomic_numbers.long()
        data.natoms = data.num_nodes

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()

        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.pos = data.pos.float()

        cutoff = 19
        if 'pos' in self.config.normalization:
            pos_std = self.config.normalization['pos']['std']
            cutoff = cutoff / pos_std
            # data.pos = self.config.normalization['pos'].normalize(data.pos)
        if data.natoms > 300:
            max_neighbors = 5
        elif data.natoms > 200:
            max_neighbors = 10
        else:
            max_neighbors = 30

        data = self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(cutoff),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
            # cutoffs=Cutoffs.from_constant(12.0),
            # max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=True,
        )
        return data
