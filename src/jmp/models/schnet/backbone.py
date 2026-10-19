"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import TypedDict
from typing_extensions import override

import torch
import torch.nn as nn
from torch_geometric.nn import SchNet
from torch_scatter import scatter

# from ...utils.goc_graph import graphs_from_batch
from ..base import GraphModelMixin
from .config import SchNetBackboneConfig 


class SchNetBackboneOutput(TypedDict):
    energy: torch.Tensor | None
    forces: torch.Tensor | None
    

class SchNetBackbone(SchNet, GraphModelMixin):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    Args:
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        use_pbc_single (bool,optional):         Process batch PBC graphs one at a time
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """

    def __init__(
        self,
        config: SchNetBackboneConfig, 
        *,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        regress_forces: bool = True,
        direct_forces: bool = False,
        otf_graph: bool = False,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        readout: str = "add",
        **kwargs
    ) -> None:
        self.config = config
        self.shared_parameters: list[tuple[nn.Parameter, int]] = []
        print("Unrecognized arguments: ", kwargs.keys())
        
        self.num_targets = 1
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        self.reduce = readout
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )
        
        

    # @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        

        # print(f"\n\n\n{data=}\n\n")
        # print(f"\n{graph=}\n\n\n")

        if self.use_pbc:
            graph = self.generate_graph(data)
            edge_attr = self.distance_expansion(graph.edge_distance)

            h = self.embedding(z)
            for interaction in self.interactions:
                h = h + interaction(h, graph.edge_index, graph.edge_distance, edge_attr)

            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            batch = torch.zeros_like(z) if batch is None else batch
            energy = scatter(h, batch, dim=0, reduce=self.reduce)
        else:
            energy = super().forward(z, pos, batch)
        return energy

    @override
    def forward(self, data):
        if self.regress_forces and not self.direct_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

        if self.regress_forces:
            assert not self.direct_forces, "Direct forces not supported"
            forces = (
                -1
                * (
                    torch.autograd.grad(
                        energy,
                        data.pos,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=True,
                    )[0]
                )
            )
        else:
            forces = None


        out: SchNetBackboneOutput = {
            "energy": energy,
            "forces": forces,
        }
        return out


    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())





