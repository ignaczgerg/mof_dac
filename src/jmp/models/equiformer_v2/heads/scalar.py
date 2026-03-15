"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch_scatter import scatter

from jmp.models.equiformer_v2 import gp_utils
from jmp.models.base import GraphData, HeadInterface
from jmp.models.equiformer_v2.transformer_block import FeedForwardNetwork
from jmp.models.equiformer_v2.weight_initialization import eqv2_init_weights

if TYPE_CHECKING:
    from torch_geometric.data import Batch


class EqV2ScalarHead(nn.Module, HeadInterface):
    def __init__(self, backbone, 
                 hidden_channels_override: int = None, 
                 output_name: str = "energy", 
                 reduce: str = "sum", 
                 output_channels: int = 1):
        super().__init__()
        self.output_name = output_name
        self.reduce = reduce
        self.output_channels = output_channels
        self.avg_num_nodes = backbone.avg_num_nodes
        hidden_channels = backbone.ffn_hidden_channels
        if hidden_channels_override is not None:
            hidden_channels = hidden_channels_override
        self.energy_block = FeedForwardNetwork(
            backbone.sphere_channels,
            hidden_channels,
            output_channels,
            backbone.lmax_list,
            backbone.mmax_list,
            backbone.SO3_grid,
            backbone.ffn_activation,
            backbone.use_gate_act,
            backbone.use_grid_mlp,
            backbone.use_sep_s2_act,
        )
        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, data: Batch, emb: dict[str, torch.Tensor | GraphData]):
        node_output = self.energy_block(emb["node_embedding"])
        node_output = node_output.embedding[:, 0, :self.output_channels]
        if gp_utils.initialized():
            node_output = gp_utils.gather_from_model_parallel_region(node_output, dim=0)

        num_graphs = len(data.natoms)

        if self.output_channels > 1:
            output = torch.zeros(
                num_graphs, self.output_channels,
                device=node_output.device,
                dtype=node_output.dtype,
            )
            for c in range(self.output_channels):
                output[:, c].index_add_(0, data.batch, node_output[:, c])
        else:
            node_output = node_output.squeeze(-1)
            output = torch.zeros(
                num_graphs,
                device=node_output.device,
                dtype=node_output.dtype,
            )
            output.index_add_(0, data.batch, node_output)

        if self.reduce == "sum":
            return {self.output_name: output / self.avg_num_nodes}
        elif self.reduce == "mean":
            if self.output_channels > 1:
                return {self.output_name: output / data.natoms.unsqueeze(-1)}
            return {self.output_name: output / data.natoms}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )

class EqV2ClassificationHead(nn.Module, HeadInterface):
    def __init__(
        self,
        backbone,
        num_classes: int,
        hidden_channels_override: int | None = None,
        output_name: str = "logits",
        reduce: str = "sum",   
    ):
        super().__init__()
        self.output_name = output_name
        self.num_classes = num_classes
        self.reduce = reduce

        hidden_channels = (
            hidden_channels_override
            if hidden_channels_override is not None
            else backbone.ffn_hidden_channels
        )

        self.class_block = FeedForwardNetwork(
            sphere_channels= backbone.sphere_channels,
            hidden_channels= hidden_channels,
            output_channels= num_classes,
            lmax_list= backbone.lmax_list,
            mmax_list= backbone.mmax_list,
            SO3_grid= backbone.SO3_grid,
            activation= backbone.ffn_activation,
            use_gate_act= backbone.use_gate_act,
            use_grid_mlp= backbone.use_grid_mlp,
            use_sep_s2_act= backbone.use_sep_s2_act,
        )

        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, data: Batch, emb: dict[str, torch.Tensor | GraphData]):
        node_rep = self.class_block(emb["node_embedding"])

        # Here we grab the first `num_classes` scalars (l=0) as logits
        # (N_nodes, lmax, num_classes) -> (N_nodes, num_classes)
        node_logits = node_rep.embedding[:, 0, :self.num_classes]  

        if gp_utils.initialized():
            node_logits = gp_utils.gather_from_model_parallel_region(
                node_logits, dim=0
            )

        # pool to per‐graph logits
        graph_logits = scatter(
            node_logits,             # [N_nodes, num_classes]
            data.batch,              # [N_nodes] graph‐idx for each node
            dim=0,
            reduce=self.reduce      # "mean" or "sum"
        )
        # graph_logits.shape == [batch_size, num_classes]

        return {self.output_name: graph_logits}
    
    
class EqV2NodeScalarHead(nn.Module, HeadInterface):
    def __init__(self, backbone, 
                 hidden_channels_override: int = None, 
                 output_name: str = "energy", 
                 reduce: str = "sum", 
                 output_channels: int = 1):
        super().__init__()
        self.output_name = output_name
        self.reduce = reduce
        self.output_channels = output_channels
        self.avg_num_nodes = backbone.avg_num_nodes
        hidden_channels = backbone.ffn_hidden_channels
        if hidden_channels_override is not None:
            hidden_channels = hidden_channels_override
        self.scalar_block = FeedForwardNetwork(
            backbone.sphere_channels,
            hidden_channels,
            output_channels,
            backbone.lmax_list,
            backbone.mmax_list,
            backbone.SO3_grid,
            backbone.ffn_activation,
            backbone.use_gate_act,
            backbone.use_grid_mlp,
            backbone.use_sep_s2_act,
        )
        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, data: Batch, emb: dict[str, torch.Tensor | GraphData]):
        output = self.scalar_block(emb["node_embedding"])
        output = output.embedding.narrow(1, 0, 1)
        output = output.view(-1, self.output_channels).contiguous()
        if gp_utils.initialized():
            output = gp_utils.gather_from_model_parallel_region(output, dim=0)
        return {self.output_name: output}