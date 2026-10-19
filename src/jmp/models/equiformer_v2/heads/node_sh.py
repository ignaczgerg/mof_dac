from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn

from jmp.models.equiformer_v2 import gp_utils
from jmp.models.base import GraphData, HeadInterface
from jmp.models.equiformer_v2.transformer_block import FeedForwardNetwork
from jmp.models.equiformer_v2.weight_initialization import eqv2_init_weights

if TYPE_CHECKING:
    from torch_geometric.data import Batch


class EqV2NodeSHHead(nn.Module, HeadInterface):
    def __init__(
        self,
        backbone,
        lmax_target: int = 4,
        hidden_channels_override: int | None = None,
        output_name: str = "co2_sh_coeffs",
    ):
        super().__init__()
        self.output_name = output_name
        self.lmax_target = lmax_target
        self.n_sh = (lmax_target + 1) ** 2
        hidden_channels = backbone.ffn_hidden_channels
        if hidden_channels_override is not None:
            hidden_channels = hidden_channels_override
        self.sh_block = FeedForwardNetwork(
            backbone.sphere_channels,
            hidden_channels,
            1,
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
        node_output = self.sh_block(emb["node_embedding"])
        output = node_output.embedding[:, : self.n_sh, 0]
        if gp_utils.initialized():
            output = gp_utils.gather_from_model_parallel_region(output, dim=0)
        return {self.output_name: output}
