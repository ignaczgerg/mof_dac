"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
import argparse
import contextlib
import logging
import typing
from functools import partial

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing_extensions import deprecated

from jmp.models.equiformer_v2 import gp_utils
# from jmp.fairchem.core.common.registry import registry
from .utils.grad import conditional_grad
from ..base import (
    GraphModelMixin,
    GraphData
)
from jmp.models.equiformer_v2.config import EquiformerV2Config
from jmp.models.equiformer_v2.heads import (
    EqV2ScalarHead, EqV2ClassificationHead, EqV2VectorHead, EqV2NodeScalarHead
    )
from .utils.smearing import (
    GaussianSmearing,
    CoulombSturmianSmearing,
    SphericalBesselSmearing,
    HankelSpectralSmearing,
    LaplaceMixSmearing,
    RadialMLP,
    ChebyshevRadialOperator,
)

from .radial_function import RadialFunction
from .edge_rot_mat import init_edge_rot_mat
from .gaussian_rbf import GaussianRadialBasisLayer
from .input_block import EdgeDegreeEmbedding
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .module_list import ModuleListInfo
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .transformer_block import (
    TransBlockV2, FeedForwardNetwork
)
from .weight_initialization import eqv2_init_weights

with contextlib.suppress(ImportError):
    pass

if typing.TYPE_CHECKING:
    from torch_geometric.data.batch import Batch

# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100


@deprecated(
    "equiformer_v2_force_head (EquiformerV2ForceHead) class is deprecated in favor of equiformerV2_rank1_head  (EqV2Rank1Head)"
)
# @registry.register_model("equiformer_v2_force_head")
class EquiformerV2ForceHead(EqV2VectorHead):
    def __init__(self, backbone, hidden_channels_override: int = None):
        logging.warning(
            "equiformerV2_force_head (EquiformerV2ForceHead) class is deprecated in favor of equiformerV2_rank1_head  (EqV2Rank1Head)"
        )
        super().__init__(backbone, hidden_channels_override=hidden_channels_override)


@deprecated(
    "equiformer_v2_energy_head (EquiformerV2EnergyHead) class is deprecated in favor of equiformerV2_scalar_head  (EqV2ScalarHead)"
)
# @registry.register_model("equiformer_v2_energy_head")
class EquiformerV2EnergyHead(EqV2ScalarHead):
    def __init__(self, backbone, hidden_channels_override: int = None, reduce: str = "sum"):
        logging.warning(
            "equiformerV2_energy_head (EquiformerV2EnergyHead) class is deprecated in favor of equiformerV2_scalar_head  (EqV2ScalarHead)"
        )
        super().__init__(backbone, hidden_channels_override=hidden_channels_override, reduce=reduce)

class ScalarRBFEncoder(nn.Module):
    """
    Self-normalizing RBF encoder for 1D descriptors.

    - Normalizes by mean |x| per batch so it's roughly scale-invariant.
    - Centers and gamma are learnable, so it adapts during training.
    """
    def __init__(self, num_rbf: int = 32):
        super().__init__()
        self.num_rbf = num_rbf
        # Start with centers in [-1, 1]; they will adapt
        centers = torch.linspace(-1.0, 1.0, num_rbf)
        self.centers = nn.Parameter(centers)         # [num_rbf]
        self.log_gamma = nn.Parameter(torch.zeros(1))  # gamma = exp(log_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B] or [B, 1] raw descriptor (e.g. PLD in Å)

        returns: [B, num_rbf]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B, 1]

        # Self-normalize: make typical magnitude ~1 without any dataset-wide stats
        scale = x.abs().mean().clamp_min(1e-6)
        x_norm = x / scale  # [B, 1]

        # RBF expansion
        # centers: [num_rbf] -> [1, num_rbf] for broadcasting
        c = self.centers.view(1, -1)       # [1, num_rbf]
        diff = x_norm - c                  # [B, num_rbf]
        gamma = torch.exp(self.log_gamma)  # scalar > 0

        rbf = torch.exp(-gamma * diff.pow(2))  # [B, num_rbf]
        return rbf

# @registry.register_model("equiformer_v2_backbone")
class EquiformerV2Backbone(nn.Module, GraphModelMixin):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        config (EquiformerV2Config): Configuration object containing model parameters
        use_pbc (bool):         Use periodic boundary conditions
        use_pbc_single (bool):         Process batch PBC graphs one at a time
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
        enforce_max_neighbors_strictly (bool):      When edges are subselected based on the `max_neighbors` arg, arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.
        avg_num_nodes (float):      Average number of nodes per graph
        avg_degree (float):         Average degree of nodes in the graph

        use_energy_lin_ref (bool):  Whether to add the per-atom energy references during prediction.
                                    During training and validation, this should be kept `False` since we use the `lin_ref` parameter in the OC22 dataloader to subtract the per-atom linear references from the energy targets.
                                    During prediction (where we don't have energy targets), this can be set to `True` to add the per-atom linear references to the predicted energies.
        load_energy_lin_ref (bool): Whether to add nn.Parameters for the per-element energy references.
                                    This additional flag is there to ensure compatibility when strict-loading checkpoints, since the `use_energy_lin_ref` flag can be either True or False even if the model is trained with linear references.
                                    You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine.
    """

    def __init__(
        self,
        config: EquiformerV2Config,
        *,
        num_targets: int = 1,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        regress_forces: bool = True,
        direct_forces: bool = False,
        otf_graph: bool = True,
        max_neighbors: int = 20,
        max_radius: float = 12.0,
        max_num_elements: int = 90,
        num_layers: int = 8,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 64,
        num_heads: int = 8,
        attn_alpha_channels: int = 64,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 128,
        norm_type: str = "layer_norm_sh",
        lmax_list: list[int] | None = [4],
        mmax_list: list[int] | None = [2],
        grid_resolution: int | None = 18,
        num_sphere_samples: int = 128,
        edge_channels: int = 128,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        use_m_share_rad: bool = False,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        attn_activation: str = "silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = True,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.1,
        proj_drop: float = 0.0,
        weight_init: str = "uniform",
        enforce_max_neighbors_strictly: bool = True,
        avg_num_nodes: float | None = None,
        avg_degree: float | None = None,
        use_energy_lin_ref: bool | None = False,
        load_energy_lin_ref: bool | None = False,
        activation_checkpoint: bool | None = False,
        # args: argparse.Namespace,
        **kwargs,
    ):
        if mmax_list is None:
            mmax_list = [2]
        if lmax_list is None:
            lmax_list = [6]
        super().__init__()

        import sys

        if "e3nn" not in sys.modules:
            logging.error("You need to install e3nn==0.4.4 to use EquiformerV2.")
            raise ImportError
        
        self.shared_parameters: list[tuple[nn.Parameter, int]] = []
        print("Unrecognized arguments: ", kwargs.keys())
        self.config = config
        # self.args = args
        self.num_targets = num_targets
        
        self.activation_checkpoint = activation_checkpoint
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.avg_num_nodes = avg_num_nodes or _AVG_NUM_NODES
        self.avg_degree = avg_degree or _AVG_DEGREE

        self.use_energy_lin_ref = use_energy_lin_ref
        self.load_energy_lin_ref = load_energy_lin_ref
        assert not (
            self.use_energy_lin_ref and not self.load_energy_lin_ref
        ), "You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine."

        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]

        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly

        self.device = "cpu"  # torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels_all: int = self.num_resolutions * self.sphere_channels

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian", "coulomb_sturmian", "cs", "bessel", "hankel", "laplace", "radial_mlp", "cheby"
        ]
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0, self.cutoff, self.num_distance_basis, 2.0
            )

        elif self.distance_function in ("coulomb_sturmian", "cs"):
            self.distance_expansion = CoulombSturmianSmearing(
                num_radial=self.num_distance_basis,
                r_cut=self.cutoff,
                p=1.0,                 # cusp-like; try p∈{1,2}
                a=None,                # ℓ-agnostic; leave None
                alpha_min=1.0 / self.cutoff,
                alpha_max=16.0 / self.cutoff,
                learn_alpha=True,
                normalize=True,
                envelope="cos",
            )
        
        elif self.distance_function == "bessel":
            self.distance_expansion = SphericalBesselSmearing(
                num_basis=self.num_distance_basis, r_cut=self.cutoff, learn_k=False
            )

        elif self.distance_function == "hankel":
            self.distance_expansion = HankelSpectralSmearing(
                num_output=self.num_distance_basis, r_cut=self.cutoff, num_modes=64
            )

        elif self.distance_function == "laplace":
            self.distance_expansion = LaplaceMixSmearing(
                num_output=self.num_distance_basis, r_cut=self.cutoff, num_mix=8, p=1.0
            )

        elif self.distance_function == "radial_mlp":
            seed = GaussianSmearing(0.0, self.cutoff, max(128, self.num_distance_basis), 2.0)
            self.distance_expansion = RadialMLP(
                seed=seed, r_cut=self.cutoff, proj_out=self.num_distance_basis, hidden=128, layers=2
            )

        elif self.distance_function == "cheby":
            self.distance_expansion = ChebyshevRadialOperator(
                num_output=self.num_distance_basis, r_cut=self.cutoff, grid_size=128, modes=64, layers=1
            )
            
        else:
            raise ValueError(f"unknown distance_function={self.distance_function}")


        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [
            self.edge_channels
        ] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo(
            f"({max(self.lmax_list)}, {max(self.lmax_list)})"
        )
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=self.grid_resolution,
                        normalization="component",
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self.avg_degree,
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
            )
            self.blocks.append(block)

        # Output blocks for energy and forces
        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels,
        )
        if self.load_energy_lin_ref:
            self.energy_lin_ref = nn.Parameter(
                torch.zeros(self.max_num_elements),
                requires_grad=False,
            )

        # ----------------------------------
        # EARLY FUSION
        # ----------------------------------
        self.mof_rbf_encoder = None
        self.mof_embedding_proj = None
        
        fusion_cfg = getattr(config, "mof_feature_fusion", None)
        if fusion_cfg and fusion_cfg.get("enabled", False) and fusion_cfg.get("fusion_type") == "early":
            num_rbf = int(fusion_cfg.get("hidden_dim"))

            # Scalar → RBF
            self.mof_rbf_encoder = ScalarRBFEncoder(num_rbf=num_rbf)

            # RBF → sphere_channels (same output dim as L=0 embedding)
            self.mof_embedding_proj = nn.Sequential(
                nn.Linear(num_rbf, self.sphere_channels),
                nn.SiLU(),
                nn.Linear(self.sphere_channels, self.sphere_channels),
            )
        # ------------------------

        self.apply(partial(eqv2_init_weights, weight_init=self.weight_init))

    @conditional_grad(torch.enable_grad())
    def forward(self, data) -> dict[str, torch.Tensor]:
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        atomic_numbers = data.atomic_numbers.long()
        # if atomic_numbers.max().item() >= self.max_num_elements:
            # print("Skipping sample with atomic number exceeding the maximum")
            # raise ValueError(
            #     f"Sample contains atomic number {atomic_numbers.max().item()} "
            #     f"which exceeds the maximum allowed {self.max_num_elements}"
            # )
        assert (
            atomic_numbers.max().item() < self.max_num_elements
        ), f"Atomic number {atomic_numbers.max().item()} exceeds that given in model config {self.max_num_elements}"
        # graph = self.generate_graph(
        #     data,
        #     enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        # )

        graph = GraphData(
            edge_index=data.main_edge_index,
            edge_distance=data.main_distance,
            edge_distance_vec=data.main_vector,
            cell_offsets=data.main_cell_offset,
            offset_distances=data.main_cell_offsets_distances,
            neighbors=data.main_num_neighbors,
            node_offset=0,
            batch_full=data.batch,
            atomic_numbers_full=data.atomic_numbers.long(),
        )

        data_batch = data.batch
        if gp_utils.initialized():
            (
                atomic_numbers,
                data_batch,
                node_offset,
                edge_index,
                edge_distance,
                edge_distance_vec,
            ) = self._init_gp_partitions(
                graph.atomic_numbers_full,
                graph.batch_full,
                graph.edge_index,
                graph.edge_distance,
                graph.edge_distance_vec,
            )
            graph.node_offset = node_offset
            graph.edge_index = edge_index
            graph.edge_distance = edge_distance
            graph.edge_distance_vec = edge_distance_vec

        ###############################################################
        # Entering Graph Parallel Region
        # after this point, if using gp, then node, edge tensors are split
        # across the graph parallel ranks, some full tensors such as
        # atomic_numbers_full are required because we need to index into the
        # full graph when computing edge embeddings or reducing nodes from neighbors
        #
        # all tensors that do not have the suffix "_full" refer to the partial tensors.
        # if not using gp, the full values are equal to the partial values
        # ie: atomic_numbers_full == atomic_numbers
        ###############################################################

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = SO3_Embedding(
            len(atomic_numbers),
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        graph.edge_distance = self.distance_expansion(graph.edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = graph.atomic_numbers_full[
                graph.edge_index[0]
            ]  # Source atom atomic number
            target_element = graph.atomic_numbers_full[
                graph.edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            graph.edge_distance = torch.cat(
                (graph.edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            graph.atomic_numbers_full,
            graph.edge_distance,
            graph.edge_index,
            len(atomic_numbers),
            graph.node_offset,
        )
        x.embedding = x.embedding + edge_degree.embedding

        # ----------------------------------
        # EARLY FUSION
        # ----------------------------------
        if (
            self.mof_rbf_encoder is not None
            and self.mof_embedding_proj is not None
            and hasattr(data, "mof_descriptor")
        ):
            # Get features: [Batch_Size, Feat_Dim]
            global_features = data.mof_descriptor

            # Scalar → RBF → fusion embedding [B, sphere_channels]
            rbf = self.mof_rbf_encoder(global_features)            # [B, num_rbf]
            global_emb = self.mof_embedding_proj(rbf)   # [B, sphere_channels]

            # Broadcast per-graph descriptor to all atoms in that graph
            #    data.batch: [N_atoms]
            atom_global_emb = global_emb[data.batch]    # [N_atoms, sphere_channels]

            # Inject into L=0 channel of the first resolution
            # x.embedding: [N_atoms, (Lmax+1)^2, sphere_channels]
            # L=0 for first resolution is index 0

            l0_index = 0
            x.embedding[:, l0_index, :] = x.embedding[:, l0_index, :] + atom_global_emb
        # --- END EARLY FUSION INJECTION 

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):
            if self.activation_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    self.blocks[i],
                    x,  # SO3_Embedding
                    graph.atomic_numbers_full,
                    graph.edge_distance,
                    graph.edge_index,
                    data_batch,  # for GraphDropPath
                    graph.node_offset,
                    use_reentrant=not self.training,
                )
            else:
                x = self.blocks[i](
                    x,  # SO3_Embedding
                    graph.atomic_numbers_full,
                    graph.edge_distance,
                    graph.edge_index,
                    batch=data_batch,  # for GraphDropPath
                    node_offset=graph.node_offset,
                )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        

        
        # feature_dict = {}
        # idx_s, idx_t = data.main_edge_index
        # feature_dict["idx_t"] = idx_t
        # feature_dict["node"] = x.embedding

        return {
            "node_embedding": x, 
            "graph": graph,
            # "feature_dict": feature_dict
        }

    def _init_gp_partitions(
        self,
        atomic_numbers_full,
        data_batch_full,
        edge_index,
        edge_distance,
        edge_distance_vec,
    ):
        """Graph Parallel
        This creates the required partial tensors for each rank given the full tensors.
        The tensors are split on the dimension along the node index using node_partition.
        """
        node_partition = gp_utils.scatter_to_model_parallel_region(
            torch.arange(len(atomic_numbers_full)).to(self.device)
        )
        edge_partition = torch.where(
            torch.logical_and(
                edge_index[1] >= node_partition.min(),
                edge_index[1] <= node_partition.max(),  # TODO: 0 or 1?
            )
        )[0]
        edge_index = edge_index[:, edge_partition]
        edge_distance = edge_distance[edge_partition]
        edge_distance_vec = edge_distance_vec[edge_partition]
        atomic_numbers = atomic_numbers_full[node_partition]
        data_batch = data_batch_full[node_partition]
        node_offset = node_partition.min().item()
        return (
            atomic_numbers,
            data_batch,
            node_offset,
            edge_index,
            edge_distance,
            edge_distance_vec,
        )

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_LinearV2,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                    GaussianRadialBasisLayer,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_LinearV2))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)



class SO3EmbeddingWithSpecialTokensPackedStacked(nn.Module):
    def __init__(self, out_dim, max_tokens=512):
        super().__init__()

        # Special token embeddings
        self.pad_token = nn.Parameter(torch.zeros(1, out_dim))    # <pad> (only if needed)
        self.bos_token = nn.Parameter(torch.randn(1, out_dim))    # <s>
        self.eos_token = nn.Parameter(torch.randn(1, out_dim))    # </s>

        self.max_tokens = max_tokens  # max tokens per packed batch

    def forward(self, data, x):
        """
        Inputs:
            data.batch: (N_atoms_total,) long tensor
            x: (N_atoms_total, D)
        Returns:
            inputs_embeds: (B_packed, L_packed, D)
            attention_mask: (B_packed, L_packed)
            packed_batch_indices: (B_packed, L_packed)
        """
        batch_indices = data.batch  # (N_atoms_total,)
        num_atoms_per_graph = torch.bincount(batch_indices)
        D = x.size(-1)

        packed_inputs = []
        packed_masks = []
        packed_batch_indices = []

        current_embeds = []
        current_batch_idx = []
        current_length = 0

        start = 0
        for i, n_atoms in enumerate(num_atoms_per_graph.tolist()):
            if n_atoms == 0:
                continue

            n_tokens = n_atoms + 2  # atoms + BOS + EOS

            # If adding this graph would exceed max_tokens, finalize current batch
            if current_length + n_tokens > self.max_tokens and current_length > 0:
                # Finalize the current batch
                inputs_embeds = torch.cat(current_embeds, dim=0)
                attention_mask = torch.ones(inputs_embeds.size(0), device=x.device, dtype=torch.long)
                batch_idx = torch.cat(current_batch_idx, dim=0)

                packed_inputs.append(inputs_embeds)
                packed_masks.append(attention_mask)
                packed_batch_indices.append(batch_idx)

                # Reset current
                current_embeds = []
                current_batch_idx = []
                current_length = 0

            # Now add the current graph
            graph_embeds = torch.cat([
                self.bos_token,        # [1, D]
                x[start:start+n_atoms],# [n_atoms, D]
                self.eos_token         # [1, D]
            ], dim=0)  # [n_atoms + 2, D]

            graph_batch_idx = x.new_full((n_tokens,), i, dtype=torch.long)

            current_embeds.append(graph_embeds)
            current_batch_idx.append(graph_batch_idx)
            current_length += n_tokens

            start += n_atoms

        # Handle any leftovers
        if current_embeds:
            inputs_embeds = torch.cat(current_embeds, dim=0)
            attention_mask = torch.ones(inputs_embeds.size(0), device=x.device, dtype=torch.long)
            batch_idx = torch.cat(current_batch_idx, dim=0)

            packed_inputs.append(inputs_embeds)
            packed_masks.append(attention_mask)
            packed_batch_indices.append(batch_idx)

        # Now pad all packed sequences to the same length (if needed)
        max_len = max(x.size(0) for x in packed_inputs)

        padded_inputs = []
        padded_masks = []
        padded_batch_indices = []

        for inputs_embeds, attention_mask, batch_idx in zip(packed_inputs, packed_masks, packed_batch_indices):
            L = inputs_embeds.size(0)
            pad_size = max_len - L

            if pad_size > 0:
                pad_embeds = self.pad_token.expand(pad_size, -1)
                pad_mask = torch.zeros(pad_size, device=x.device, dtype=torch.long)
                pad_batch_idx = x.new_full((pad_size,), -1, dtype=torch.long)  # -1 means PAD

                inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=0)
                attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
                batch_idx = torch.cat([batch_idx, pad_batch_idx], dim=0)

            padded_inputs.append(inputs_embeds.unsqueeze(0))  # [1, L, D]
            padded_masks.append(attention_mask.unsqueeze(0)) # [1, L]
            padded_batch_indices.append(batch_idx.unsqueeze(0)) # [1, L]

        # Stack into batches
        inputs_embeds = torch.cat(padded_inputs, dim=0)       # [B_packed, L_packed, D]
        attention_mask = torch.cat(padded_masks, dim=0)       # [B_packed, L_packed]
        packed_batch_indices = torch.cat(padded_batch_indices, dim=0)  # [B_packed, L_packed]

        return inputs_embeds, attention_mask, packed_batch_indices

class SO3EmbeddingWithSpecialTokens(nn.Module):
    def __init__(self, out_dim, max_tokens=512):
        super().__init__()

        # Special token embeddings
        self.pad_token = nn.Parameter(torch.zeros(1, 1, out_dim))    # <pad>
        self.bos_token = nn.Parameter(torch.randn(1, 1, out_dim))    # <s>
        self.eos_token = nn.Parameter(torch.randn(1, 1, out_dim))    # </s>

        self.max_tokens = max_tokens  # max atoms + 2 for <s>, </s>

    def forward(self, data, x):
        B = data.num_graphs
        num_atoms_per_graph = torch.bincount(data.batch)
        D = x.size(-1)

        inputs_embeds = x.new_zeros(B, self.max_tokens, D)
        attention_mask = x.new_zeros(B, self.max_tokens)

        start = 0
        for i, n_atoms in enumerate(num_atoms_per_graph.tolist()):
            # Safety padding
            n = min(n_atoms, self.max_tokens - 2)

            # Add special tokens
            inputs_embeds[i, 0] = self.bos_token
            inputs_embeds[i, 1:n+1] = x[start:start+n]
            inputs_embeds[i, n+1] = self.eos_token

            attention_mask[i, :n+2] = 1
            start += n_atoms

        # Fill remaining PAD tokens with pad_token
        pad_mask = attention_mask == 0
        inputs_embeds[pad_mask] = self.pad_token.squeeze(0).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        return inputs_embeds, attention_mask

class ExtractAtomEmbeddingsPackedStacked(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs_embeds, attention_mask, packed_batch_indices):
        """
        Fully vectorized revert of packed + stacked sequences
        Inputs:
            - inputs_embeds: Tensor[B, T, D]
            - attention_mask: Tensor[B, T]
            - packed_batch_indices: Tensor[B, T]
        Returns:
            - x: Tensor[N_atoms_total, D]
            - batch: Tensor[N_atoms_total]
        """
        B, T, D = inputs_embeds.shape

        # Flatten across batch dimension: [B * T, D]
        inputs_embeds_flat = inputs_embeds.reshape(-1, D)  # (B*T, D)
        attention_mask_flat = attention_mask.reshape(-1)   # (B*T)
        packed_batch_indices_flat = packed_batch_indices.reshape(-1)  # (B*T)

        # Find BOS (start of a new graph inside each batch)
        is_bos = torch.zeros_like(attention_mask_flat, dtype=torch.bool)
        is_bos[0] = True
        is_bos[1:] = packed_batch_indices_flat[1:] != packed_batch_indices_flat[:-1]

        # Find EOS (just before each new BOS, plus last token if necessary)
        bos_positions = torch.where(is_bos)[0]  # positions of BOS
        eos_positions = torch.cat([bos_positions[1:] - 1, attention_mask_flat.nonzero()[-1]])

        # Mask for BOS and EOS
        bos_eos_mask = torch.zeros_like(attention_mask_flat, dtype=torch.bool)
        bos_eos_mask[bos_positions] = True
        bos_eos_mask[eos_positions] = True

        # Atom mask = real token AND not BOS/EOS
        atom_mask = attention_mask_flat.bool() & (~bos_eos_mask)

        # Extract
        x = inputs_embeds_flat[atom_mask]             # (N_atoms_total, D)
        batch = packed_batch_indices_flat[atom_mask]  # (N_atoms_total)

        return x, batch

class ExtractAtomEmbeddings(nn.Module):
    def __init__(self, max_tokens=512):
        super().__init__()
        self.max_tokens = max_tokens

    def forward(self, inputs_embeds, attention_mask):
        """
        Reverse operation: (B, max_tokens, D) → (N_atoms_total, D) with atom-level index reconstruction
        Inputs:
            - inputs_embeds: Tensor[B, max_tokens, D]
            - attention_mask: Tensor[B, max_tokens], where 1 = real token, 0 = PAD
        Returns:
            - x: Tensor[N_atoms_total, D]  # only atom tokens (excludes BOS, EOS, PAD)
            - batch: Tensor[N_atoms_total]  # graph indices per atom (0 <= i < B)
        """
        B, T, D = inputs_embeds.shape
        atom_embeddings = []
        batch_index = []

        for i in range(B):
            # Valid tokens (excluding PAD), then remove BOS and EOS (always at positions 0 and n+1)
            valid_mask = attention_mask[i].bool()
            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)

            if len(valid_indices) <= 2:
                continue  # skip if graph has only BOS and EOS

            atom_start = 1
            atom_end = len(valid_indices) - 1  # exclude EOS

            atom_embeddings.append(inputs_embeds[i, atom_start:atom_end])
            batch_index.append(inputs_embeds.new_full((atom_end - atom_start,), i, dtype=torch.long))

        x = torch.cat(atom_embeddings, dim=0)      # [N_atoms_total, D]
        batch = torch.cat(batch_index, dim=0)       # [N_atoms_total]

        return x, batch

def make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Vectorized position_ids compatible with RoPE for packed padded sequences.

    Args:
        attention_mask: (B_packed, L_packed) with 1 for real tokens, 0 for padding

    Returns:
        position_ids: (B_packed, L_packed) with token positions starting at 0 per sequence
    """
    # Cumulative sum along dim=1 gives the position index + 1
    position_ids = attention_mask.cumsum(dim=1) - 1
    # Set positions to 0 for padding tokens
    position_ids = position_ids * attention_mask
    return position_ids




# @registry.register_model("autoreg_equiformer_v2")
class AutoregressiveEquiformerV2Backbone(nn.Module, GraphModelMixin):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid
        
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        
        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        
        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs. 
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """
    def __init__(
        self,
        config: EquiformerV2Config,
        *,
        num_targets: int = 1,
        use_pbc: bool = True,
        regress_forces: bool = True,
        direct_forces: bool = False,
        otf_graph: bool = True,
        max_neighbors: int = 500,
        max_radius: float = 5.0,
        max_num_elements: int = 90,

        num_layers: int = 12,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 128,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,
        
        norm_type: str ='rms_norm_sh',
        
        lmax_list: list[int] = [6],
        mmax_list: list[int] = [2],
        grid_resolution: int | None = None, 

        num_sphere_samples: int = 128,

        edge_channels=128,
        use_atom_edge_embedding=True, 
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512, 

        attn_activation: str = 'scaled_silu',
        use_s2_act_attn: bool = False, 
        use_attn_renorm: bool = True,
        ffn_activation: str ='scaled_silu',
        use_gate_act:   bool = False,
        use_grid_mlp:   bool = False, 
        use_sep_s2_act: bool = True,

        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05, 
        proj_drop: float = 0.0, 

        enforce_max_neighbors_strictly: bool = True,
        avg_num_nodes: float | None = None,
        avg_degree: float | None = None,
        use_energy_lin_ref: bool | None = False,
        load_energy_lin_ref: bool | None = False,
        activation_checkpoint: bool | None = False,


        weight_init: str = 'normal',
        
        # args: argparse.Namespace,
        **kwargs,
    ):
        super().__init__()

        import sys

        if "e3nn" not in sys.modules:
            logging.error("You need to install e3nn==0.4.4 to use EquiformerV2.")
            raise ImportError
        
        self.shared_parameters: list[tuple[nn.Parameter, int]] = []
        print("Unrecognized arguments: ", kwargs.keys())
        self.config = config
        # self.args = args
        self.num_targets = num_targets
        
        self.activation_checkpoint = activation_checkpoint
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding 
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.avg_num_nodes = avg_num_nodes or _AVG_NUM_NODES
        self.avg_degree = avg_degree or _AVG_DEGREE

        self.use_energy_lin_ref = use_energy_lin_ref
        self.load_energy_lin_ref = load_energy_lin_ref
        assert not (
            self.use_energy_lin_ref and not self.load_energy_lin_ref
        ), "You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine."


        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.device = 'cpu' #torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        
        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        
        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
            #self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self.avg_degree
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop,
            )
            self.blocks.append(block)
        
        # Output blocks for energy and forces
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)

        # self.coord_head = FeedForwardNetwork(self.sphere_channels,
        #         self.ffn_hidden_channels, 
        #         3,
        #         self.lmax_list,
        #         self.mmax_list,
        #         self.SO3_grid,  
        #         self.ffn_activation,
        #         self.use_gate_act,
        #         self.use_grid_mlp,
        #         self.use_sep_s2_act
        #     )
        # self.atomic_num_head = FeedForwardNetwork(self.sphere_channels,
        #     self.ffn_hidden_channels, 
        #     self.max_num_elements,
        #     self.lmax_list,
        #     self.mmax_list,
        #     self.SO3_grid,  
        #     self.ffn_activation,
        #     self.use_gate_act,
        #     self.use_grid_mlp,
        #     self.use_sep_s2_act
        # )
            
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        atomic_numbers = data.atomic_numbers[:, 0].long() # slicing to handle task idx
        # if atomic_numbers.max().item() >= self.max_num_elements:
            # print("Skipping sample with atomic number exceeding the maximum")
            # raise ValueError(
            #     f"Sample contains atomic number {atomic_numbers.max().item()} "
            #     f"which exceeds the maximum allowed {self.max_num_elements}"
            # )
        assert (
            atomic_numbers.max().item() < self.max_num_elements
        ), f"Atomic number {atomic_numbers.max().item()} exceeds that given in model config {self.max_num_elements}"
        # graph = self.generate_graph(
        #     data,
        #     enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        # )

        graph = GraphData(
            edge_index=data.main_edge_index,
            edge_distance=data.main_distance,
            edge_distance_vec=data.main_vector,
            cell_offsets=data.main_cell_offset,
            offset_distances=data.main_cell_offsets_distances,
            neighbors=data.main_num_neighbors,
            node_offset=0,
            batch_full=data.batch,
            atomic_numbers_full=atomic_numbers,
        )

        data_batch = data.batch
        if gp_utils.initialized():
            (
                atomic_numbers,
                data_batch,
                node_offset,
                edge_index,
                edge_distance,
                edge_distance_vec,
            ) = self._init_gp_partitions(
                graph.atomic_numbers_full,
                graph.batch_full,
                graph.edge_index,
                graph.edge_distance,
                graph.edge_distance_vec,
            )
            graph.node_offset = node_offset
            graph.edge_index = edge_index
            graph.edge_distance = edge_distance
            graph.edge_distance_vec = edge_distance_vec

        ###############################################################
        # Entering Graph Parallel Region
        # after this point, if using gp, then node, edge tensors are split
        # across the graph parallel ranks, some full tensors such as
        # atomic_numbers_full are required because we need to index into the
        # full graph when computing edge embeddings or reducing nodes from neighbors
        #
        # all tensors that do not have the suffix "_full" refer to the partial tensors.
        # if not using gp, the full values are equal to the partial values
        # ie: atomic_numbers_full == atomic_numbers
        ###############################################################

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # # Compute 3x3 rotation matrix per edge
        # edge_rot_mat = self._init_edge_rot_mat(
        #     data, graph.edge_index, graph.edge_distance_vec
        # )

        # # Initialize the WignerD matrices and other values for spherical harmonic calculations
        # for i in range(self.num_resolutions):
        #     self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = SO3_Embedding(
            len(atomic_numbers),
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        graph.edge_distance = self.distance_expansion(graph.edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = graph.atomic_numbers_full[
                graph.edge_index[0]
            ]  # Source atom atomic number
            target_element = graph.atomic_numbers_full[
                graph.edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            graph.edge_distance = torch.cat(
                (graph.edge_distance, source_embedding, target_embedding), dim=1
            )
        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            graph.atomic_numbers_full,
            graph.edge_distance,
            graph.edge_index,
            len(atomic_numbers),
            graph.node_offset,
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        
        ###### Causal Mask ######
        # Define the generation order
        # We need to mask future nodes for each edge 
        row, col = graph.edge_index
        causal_mask = row > col  # Only allow attention from a node to itself or previous nodes
        # TODO: possibly support activation_checkpoint (check the base EquiformerV2Backbone)
        for i in range(self.num_layers):
            x = self.blocks[i](
                x,                  # SO3_Embedding
                graph.atomic_numbers_full,
                graph.edge_distance,
                graph.edge_index,
                batch=data.batch,    # for GraphDropPath
                causal_mask=causal_mask,      # for causal_mask
            )
        
        # Final layer norm
        x.embedding = self.norm(x.embedding)
        ###############################################################
        # Atomic number Estimation
        ###############################################################
        # nodes_atom_num = self.atomic_num_head(x)
        # nodes_atom_num = nodes_atom_num.embedding.narrow(1, 0, 1)
        # nodes_atom_num = nodes_atom_num.reshape(-1, self.max_num_elements)

        # # Using regression head to predict the atom positions
        # nodes_positions = self.coord_head(x)
        # nodes_positions = nodes_positions.embedding.narrow(1, 0, 1)
        # nodes_positions = nodes_positions.reshape(-1, 3)

        return {
            "node_embedding": x, 
            "graph": graph,
            "causal_mask": causal_mask,
            # "nodes_positions": nodes_positions,
            # "nodes_atom_num": nodes_atom_num,
            # "feature_dict": feature_dict
        }  
        # ###############################################################
        # # Atom Position Estimation
        # ###############################################################
        # nodes_positions = self.coord_head(x)
        # nodes_positions = nodes_positions.embedding.narrow(1, 0, 1)
        # nodes_positions = nodes_positions.reshape(-1, 3)

        # ###############################################################
        # # Atomic number Estimation
        # ###############################################################
        # nodes_atom_num = self.atomic_num_head(x)
        # nodes_atom_num = nodes_atom_num.embedding.narrow(1, 0, 1)
        # nodes_atom_num = nodes_atom_num.reshape(-1, self.max_num_elements)

        # ###############################################################
        # # Energy estimation
        # ###############################################################
        # node_energy = self.energy_block(x) 
        # node_energy = node_energy.embedding.narrow(1, 0, 1)
        # energy = torch.zeros(len(data.natoms), device=node_energy.device, dtype=node_energy.dtype)
        # energy.index_add_(0, data.batch, node_energy.view(-1))
        # energy = energy / _AVG_NUM_NODES
        # ###############################################################
        # # Force estimation
        # ###############################################################
        # if self.regress_forces:
        #     forces = self.force_block(x,
        #         atomic_numbers,
        #         edge_distance,
        #         edge_index)
        #     forces = forces.embedding.narrow(1, 1, 3)
        #     forces = forces.view(-1, 3)            
        # if not self.regress_forces:
        #     return energy, nodes_positions, nodes_atom_num
        # else:
        #     return energy, forces, nodes_positions, nodes_atom_num


    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
        

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)
    


# @registry.register_model("autoreg_llama_equiformer_v2")
class AutoregressiveLlamaEquiformerV2Backbone(nn.Module, GraphModelMixin):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid
        
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        
        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        
        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs. 
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """
    def __init__(
        self,
        config: EquiformerV2Config,
        *,
        num_targets: int = 1,
        use_pbc: bool = True,
        regress_forces: bool = True,
        direct_forces: bool = False,
        otf_graph: bool = True,
        max_neighbors: int = 500,
        max_radius: float = 5.0,
        max_num_elements: int = 90,

        num_layers: int = 12,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 128,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,
        
        norm_type: str ='rms_norm_sh',
        
        lmax_list: list[int] = [6],
        mmax_list: list[int] = [2],
        grid_resolution: int | None = None, 

        num_sphere_samples: int = 128,

        edge_channels=128,
        use_atom_edge_embedding=True, 
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512, 

        attn_activation: str = 'scaled_silu',
        use_s2_act_attn: bool = False, 
        use_attn_renorm: bool = True,
        ffn_activation: str ='scaled_silu',
        use_gate_act:   bool = False,
        use_grid_mlp:   bool = False, 
        use_sep_s2_act: bool = True,

        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05, 
        proj_drop: float = 0.0, 

        enforce_max_neighbors_strictly: bool = True,
        avg_num_nodes: float | None = None,
        avg_degree: float | None = None,
        use_energy_lin_ref: bool | None = False,
        load_energy_lin_ref: bool | None = False,
        activation_checkpoint: bool | None = False,


        weight_init: str = 'normal',
        llm_checkpoint: str = "HuggingFaceTB/SmolLM2-1.7B",
        use_position_enc: bool = False,
        max_tokens=None,
        add_prop_pred=False,
        init_rand_special_tokens=True,

        # args: argparse.Namespace,
        **kwargs,
    ):
        super().__init__()

        import sys

        if "e3nn" not in sys.modules:
            logging.error("You need to install e3nn==0.4.4 to use EquiformerV2.")
            raise ImportError
        
        self.shared_parameters: list[tuple[nn.Parameter, int]] = []
        print("Unrecognized arguments: ", kwargs.keys())
        self.config = config
        # self.args = args
        self.num_targets = num_targets

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding 
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.avg_num_nodes = avg_num_nodes or _AVG_NUM_NODES
        self.avg_degree = avg_degree or _AVG_DEGREE

        self.use_energy_lin_ref = use_energy_lin_ref
        self.load_energy_lin_ref = load_energy_lin_ref
        assert not (
            self.use_energy_lin_ref and not self.load_energy_lin_ref
        ), "You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine."


        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.device = 'cpu' #torch.cuda.current_device()
        self.add_prop_pred = add_prop_pred

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        
        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        
        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
            #self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [
            self.edge_channels
        ] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding (check this)
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self.avg_degree
        ) 

        self.use_position_enc = use_position_enc
        # try:
        #     self.model = AutoModelForCausalLM.from_pretrained(llm_checkpoint, device_map="auto", attn_implementation="flash_attention_2")
        #     print(f'############## Loading LLM: {llm_checkpoint} with Flash attention 2 ##############')
        # except Exception as e:
        #     self.model = AutoModelForCausalLM.from_pretrained(llm_checkpoint)
        #     print(f'############## Loading LLM: {llm_checkpoint} without Flash attention 2 ##############')
        
        # self.llm_config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
        # self.model =  AutoModelForCausalLM(self.llm_config)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_checkpoint,
                device_map="auto",
                attn_implementation="flash_attention_2",
            ).to(self.device)
            print(f'############## Loading LLM: {llm_checkpoint} with Flash attention 2 ##############')
        except Exception as e:
            print("Exception in llm init: ", e)
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_checkpoint,
                # device_map="auto",
                attn_implementation="flash_attention_2",
            ).to(self.device)
            print(f'############## Loading LLM: {llm_checkpoint} without Flash attention 2 ##############')
            
        
        input_dim = self.model.config.hidden_size
        self.so3_input_dim = self.sphere_channels * (self.lmax_list[0] + 1) ** 2 
        self.proj = nn.Sequential(
            nn.LayerNorm(self.so3_input_dim),
            nn.Linear(self.so3_input_dim, input_dim)
            )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.so3_input_dim)
            )
        # max_tokens = self.model.config.max_position_embeddings
        # max_tokens = 512
        if max_tokens is None:
            max_tokens = self.model.config.max_position_embeddings
        print(f'############## Max tokens LLM: {max_tokens} ##############')
        # self.LLMfromS03Embedding=SO3EmbeddingWithSpecialTokens(input_dim, max_tokens=max_tokens)   
        self.LLMfromS03Embedding=SO3EmbeddingWithSpecialTokensPackedStacked(input_dim, max_tokens=max_tokens)
        if not init_rand_special_tokens:
            tokenizer = AutoTokenizer.from_pretrained(llm_checkpoint)
            embedding_layer = self.model.get_input_embeddings()  # nn.Embedding
            # Get token IDs for BOS and EOS
            bos_id = tokenizer.bos_token_id  # usually 1
            eos_id = tokenizer.eos_token_id  # usually 2
            # Extract embeddings
            bos_embedding = embedding_layer(torch.tensor([bos_id], device=embedding_layer.weight.device))
            eos_embedding = embedding_layer(torch.tensor([eos_id], device=embedding_layer.weight.device))
            # bos_embedding = embedding_layer(torch.tensor([bos_id])).to(self.LLMfromS03Embedding.bos_token.device)  # shape [1, D]
            # eos_embedding = embedding_layer(torch.tensor([eos_id])).to(self.LLMfromS03Embedding.eos_token.device)  # shape [1, D]
            # Assign to special tokens
            with torch.no_grad():
                self.LLMfromS03Embedding.bos_token.data.copy_(bos_embedding)
                self.LLMfromS03Embedding.eos_token.data.copy_(eos_embedding)
            # with torch.no_grad():
            #     self.LLMfromS03Embedding.bos_token.copy_(bos_embedding)
            #     self.LLMfromS03Embedding.eos_token.copy_(eos_embedding)
            # # Keep them frozen
            # self.LLMfromS03Embedding.bos_token.requires_grad = False
            # self.LLMfromS03Embedding.eos_token.requires_grad = False

        self.ExtractAtomEmbeddings=ExtractAtomEmbeddingsPackedStacked()

        
        
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)
        # self.apply(self._init_weights)  # check this because we can reinitialize the weights of the llm
        self.apply_custom_init()
        self.apply(self._uniform_init_rad_func_linear_weights)
    
    def apply_custom_init(self):
        for name, module in self.named_modules():
            if "model" in name:  # Skip the LLM
                continue
            self._init_weights(module)
        
        
        # self.energy_block = FeedForwardNetwork(
        #     self.sphere_channels,
        #     self.ffn_hidden_channels, 
        #     1,
        #     self.lmax_list,
        #     self.mmax_list,
        #     self.SO3_grid,  
        #     self.ffn_activation,
        #     self.use_gate_act,
        #     self.use_grid_mlp,
        #     self.use_sep_s2_act
        # )
        # self.coord_head = FeedForwardNetwork(self.sphere_channels,
        #     self.sphere_channels, 
        #     3,
        #     self.lmax_list,
        #     self.mmax_list,
        #     self.SO3_grid,  
        #     self.ffn_activation,
        #     self.use_gate_act,
        #     self.use_grid_mlp,
        #     self.use_sep_s2_act
        # )
        # self.atomic_num_head = FeedForwardNetwork(self.sphere_channels,
        #     self.sphere_channels, 
        #     self.max_num_elements,
        #     self.lmax_list,
        #     self.mmax_list,
        #     self.SO3_grid,  
        #     self.ffn_activation,
        #     self.use_gate_act,
        #     self.use_grid_mlp,
        #     self.use_sep_s2_act
        # )

        # if self.add_prop_pred:
        #     if self.regress_forces:
        #         self.force_block = SO2EquivariantGraphAttention(
        #             self.sphere_channels,
        #             self.attn_hidden_channels,
        #             self.num_heads, 
        #             self.attn_alpha_channels,
        #             self.attn_value_channels, 
        #             1,
        #             self.lmax_list,
        #             self.mmax_list,
        #             self.SO3_rotation, 
        #             self.mappingReduced, 
        #             self.SO3_grid, 
        #             self.max_num_elements,
        #             self.edge_channels_list,
        #             self.block_use_atom_edge_embedding, 
        #             self.use_m_share_rad,
        #             self.attn_activation, 
        #             self.use_s2_act_attn, 
        #             self.use_attn_renorm,
        #             self.use_gate_act,
        #             self.use_sep_s2_act,
        #             alpha_drop=0.0
        #         )


        #     self.energy_block = FeedForwardNetwork(
        #         self.sphere_channels,
        #         self.ffn_hidden_channels, 
        #         1,
        #         self.lmax_list,
        #         self.mmax_list,
        #         self.SO3_grid,  
        #         self.ffn_activation,
        #         self.use_gate_act,
        #         self.use_grid_mlp,
        #         self.use_sep_s2_act
        #     )

        #     if self.regress_forces:
        #         self.force_block = SO2EquivariantGraphAttention(
        #             self.sphere_channels,
        #             self.attn_hidden_channels,
        #             self.num_heads, 
        #             self.attn_alpha_channels,
        #             self.attn_value_channels, 
        #             1,
        #             self.lmax_list,
        #             self.mmax_list,
        #             self.SO3_rotation, 
        #             self.mappingReduced, 
        #             self.SO3_grid, 
        #             self.max_num_elements,
        #             self.edge_channels_list,
        #             self.block_use_atom_edge_embedding, 
        #             self.use_m_share_rad,
        #             self.attn_activation, 
        #             self.use_s2_act_attn, 
        #             self.use_attn_renorm,
        #             self.use_gate_act,
        #             self.use_sep_s2_act,
        #             alpha_drop=0.0
        #         )
            

    # def compute_graph_emb(self, data): 
    #     self.batch_size = len(data.natoms)
    #     self.dtype = data.pos.dtype
    #     self.device = data.pos.device

    #     atomic_numbers = data.atomic_numbers.long()
    #     num_atoms = len(atomic_numbers)
    #     pos = data.pos

    #     (
    #         edge_index,
    #         edge_distance,
    #         edge_distance_vec,
    #         cell_offsets,
    #         _,  # cell offset distances
    #         neighbors,
    #     ) = self.generate_graph(data)

    #     ###############################################################
    #     # Initialize data structures
    #     ###############################################################

    #     # Compute 3x3 rotation matrix per edge
    #     edge_rot_mat = self._init_edge_rot_mat(
    #         data, edge_index, edge_distance_vec
    #     )

    #     # Initialize the WignerD matrices and other values for spherical harmonic calculations
    #     for i in range(self.num_resolutions):
    #         self.SO3_rotation[i].set_wigner(edge_rot_mat)

    #     ###############################################################
    #     # Initialize node embeddings
    #     ###############################################################

    #     # Init per node representations using an atomic number based embedding
    #     offset = 0
    #     x = SO3_Embedding(
    #         num_atoms,
    #         self.lmax_list,
    #         self.sphere_channels,
    #         self.device,
    #         self.dtype,
    #     )

    #     offset_res = 0
    #     offset = 0
    #     # Initialize the l = 0, m = 0 coefficients for each resolution
    #     for i in range(self.num_resolutions):
    #         if self.num_resolutions == 1:
    #             x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
    #         else:
    #             x.embedding[:, offset_res, :] = self.sphere_embedding(
    #                 atomic_numbers
    #                 )[:, offset : offset + self.sphere_channels]
    #         offset = offset + self.sphere_channels
    #         offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

    #     # Edge encoding (distance and atom edge)
    #     edge_distance = self.distance_expansion(edge_distance)
    #     if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
    #         source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
    #         target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
    #         source_embedding = self.source_embedding(source_element)
    #         target_embedding = self.target_embedding(target_element)
    #         edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

    #     # Edge-degree embedding
    #     edge_degree = self.edge_degree_embedding(
    #         atomic_numbers,
    #         edge_distance,
    #         edge_index)
    #     x.embedding = x.embedding + edge_degree.embedding
        
    #     return x, atomic_numbers, edge_distance, edge_index

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        atomic_numbers = data.atomic_numbers.long()
        # if atomic_numbers.max().item() >= self.max_num_elements:
            # print("Skipping sample with atomic number exceeding the maximum")
            # raise ValueError(
            #     f"Sample contains atomic number {atomic_numbers.max().item()} "
            #     f"which exceeds the maximum allowed {self.max_num_elements}"
            # )
        assert (
            atomic_numbers.max().item() < self.max_num_elements
        ), f"Atomic number {atomic_numbers.max().item()} exceeds that given in model config {self.max_num_elements}"
        # graph = self.generate_graph(
        #     data,
        #     enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        # )

        graph = GraphData(
            edge_index=data.main_edge_index,
            edge_distance=data.main_distance,
            edge_distance_vec=data.main_vector,
            cell_offsets=data.main_cell_offset,
            offset_distances=data.main_cell_offsets_distances,
            neighbors=data.main_num_neighbors,
            node_offset=0,
            batch_full=data.batch,
            atomic_numbers_full=data.atomic_numbers.long(),
        )

        data_batch = data.batch
        if gp_utils.initialized():
            (
                atomic_numbers,
                data_batch,
                node_offset,
                edge_index,
                edge_distance,
                edge_distance_vec,
            ) = self._init_gp_partitions(
                graph.atomic_numbers_full,
                graph.batch_full,
                graph.edge_index,
                graph.edge_distance,
                graph.edge_distance_vec,
            )
            graph.node_offset = node_offset
            graph.edge_index = edge_index
            graph.edge_distance = edge_distance
            graph.edge_distance_vec = edge_distance_vec

        ###############################################################
        # Entering Graph Parallel Region
        # after this point, if using gp, then node, edge tensors are split
        # across the graph parallel ranks, some full tensors such as
        # atomic_numbers_full are required because we need to index into the
        # full graph when computing edge embeddings or reducing nodes from neighbors
        #
        # all tensors that do not have the suffix "_full" refer to the partial tensors.
        # if not using gp, the full values are equal to the partial values
        # ie: atomic_numbers_full == atomic_numbers
        ###############################################################

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = SO3_Embedding(
            len(atomic_numbers),
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        graph.edge_distance = self.distance_expansion(graph.edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = graph.atomic_numbers_full[
                graph.edge_index[0]
            ]  # Source atom atomic number
            target_element = graph.atomic_numbers_full[
                graph.edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            graph.edge_distance = torch.cat(
                (graph.edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            graph.atomic_numbers_full,
            graph.edge_distance,
            graph.edge_index,
            len(atomic_numbers),
            graph.node_offset,
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Compute S03 node embeddings
        ###############################################################
        # x, atomic_numbers, edge_distance, edge_index = self.compute_graph_emb(data)
        x_emb = x.embedding.view(x.embedding.size(0), -1)
        x_emb = self.proj(x_emb)
        ###############################################################
        # Parse S03 node embeddings to LLM inputs
        ###############################################################
        inputs_embeds, attention_mask, packed_batch_indices = self.LLMfromS03Embedding(data, x_emb)
        ###############################################################
        # Forward pass through LLM
        ###############################################################
        

        # with autocast():
        if not self.use_position_enc:
            ############## LLM forward pass without position encoder ##############
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, 
                                output_hidden_states=True, return_dict=True)
        else:
            ############## LLM forward pass with position encoder (RoPE) ##############
            position_ids=make_position_ids(attention_mask)
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, 
                                    position_ids=position_ids, output_hidden_states=True, return_dict=True)
        outputs_embeddings = outputs.hidden_states[-1]
        outputs_embeddings = self.out_proj(outputs_embeddings)
        ###############################################################
        # Recover the original S03 node embeddings
        # and update the spherical node embeddings
        ###############################################################
        outputs_embeddings_s03, batch = self.ExtractAtomEmbeddings(outputs_embeddings, attention_mask, packed_batch_indices)
        outputs_embeddings_s03 = outputs_embeddings_s03.view(outputs_embeddings_s03.size(0), -1, self.sphere_channels)
        data.batch = batch

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        
        outputs_embeddings_s03 = outputs_embeddings_s03.to(torch.float)
        x.embedding = self.norm(outputs_embeddings_s03)

        
        return {
            "node_embedding": x, 
            "graph": graph,
            # "feature_dict": feature_dict
        }        


        # ###############################################################
        # # Atom Position Estimation
        # ###############################################################
        # nodes_positions = self.coord_head(x)
        # nodes_positions = nodes_positions.embedding.narrow(1, 0, 1)
        # nodes_positions = nodes_positions.reshape(-1, 3)

        # ###############################################################
        # # Atomic number Estimation
        # ###############################################################
        # nodes_atom_num = self.atomic_num_head(x)
        # nodes_atom_num = nodes_atom_num.embedding.narrow(1, 0, 1)
        # nodes_atom_num = nodes_atom_num.reshape(-1, self.max_num_elements)

        # ###############################################################
        # # Energy estimation
        # ###############################################################
        # node_energy = self.energy_block(x) 
        # node_energy = node_energy.embedding.narrow(1, 0, 1)
        # energy = torch.zeros(len(data.natoms), device=node_energy.device, dtype=node_energy.dtype)
        # energy.index_add_(0, data.batch, node_energy.view(-1))
        # energy = energy / _AVG_NUM_NODES
        # ###############################################################
        # # Force estimation
        # ###############################################################
        # if self.regress_forces:
        #     forces = self.force_block(x,
        #         atomic_numbers,
        #         edge_distance,
        #         edge_index)
        #     forces = forces.embedding.narrow(1, 1, 3)
        #     forces = forces.view(-1, 3)            
        # if not self.regress_forces:
        #     return energy, nodes_positions, nodes_atom_num
        # else:
        #     return energy, forces, nodes_positions, nodes_atom_num


    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
        

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    
    def _init_gp_partitions(
        self,
        atomic_numbers_full,
        data_batch_full,
        edge_index,
        edge_distance,
        edge_distance_vec,
    ):
        """Graph Parallel
        This creates the required partial tensors for each rank given the full tensors.
        The tensors are split on the dimension along the node index using node_partition.
        """
        node_partition = gp_utils.scatter_to_model_parallel_region(
            torch.arange(len(atomic_numbers_full)).to(self.device)
        )
        edge_partition = torch.where(
            torch.logical_and(
                edge_index[1] >= node_partition.min(),
                edge_index[1] <= node_partition.max(),  # TODO: 0 or 1?
            )
        )[0]
        edge_index = edge_index[:, edge_partition]
        edge_distance = edge_distance[edge_partition]
        edge_distance_vec = edge_distance_vec[edge_partition]
        atomic_numbers = atomic_numbers_full[node_partition]
        data_batch = data_batch_full[node_partition]
        node_offset = node_partition.min().item()
        return (
            atomic_numbers,
            data_batch,
            int(node_offset),
            edge_index,
            edge_distance,
            edge_distance_vec,
        )

    # Initialize the edge rotation matrics


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            # Include all model (LLM) parameters
            if module_name.startswith("model"):
                for param_name, _ in module.named_parameters():
                    full_name = f"{module_name}.{param_name}"
                    if full_name in named_parameters_list:
                        no_wd_list.append(full_name)
                continue  # Skip further checks below to avoid duplication

            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)):

                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)

