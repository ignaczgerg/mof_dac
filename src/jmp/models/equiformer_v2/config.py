"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Literal, TypeAlias
from jmp.lightning import TypedConfig
from typing_extensions import override

EquiformerV2BackboneOutput: TypeAlias = dict[str, Any]

class EquiformerV2Config(TypedConfig):
    name: Literal["EquiformerV2", "equiformer_v2"] = "equiformer_v2"  

    num_targets: int = 1
    num_layers: int = 8
    sphere_channels: int = 128
    attn_hidden_channels: int = 64
    num_heads: int = 8
    attn_alpha_channels: int = 64
    attn_value_channels: int = 16
    ffn_hidden_channels: int = 128
    norm_type: str = "layer_norm_sh"

    lmax_list: list[int] = [4]
    mmax_list: list[int] = [2]
    grid_resolution: int = 18

    num_sphere_samples: int = 128

    edge_channels: int = 128
    use_atom_edge_embedding: bool = True
    share_atom_edge_embedding: bool = False
    use_m_share_rad: bool = False
    distance_function: Literal["gaussian","coulomb_sturmian", "cs","bessel", "hankel", "radial_mlp", "cheby", "laplace"] = "gaussian"
    num_distance_basis: int = 512

    attn_activation: str = "silu"
    use_s2_act_attn: bool = False
    use_attn_renorm: bool = True
    ffn_activation: str = "silu"
    use_gate_act: bool = False
    use_grid_mlp: bool = True
    use_sep_s2_act: bool = True

    alpha_drop: float = 0.1
    drop_path_rate: float = 0.1
    proj_drop: float = 0.0
    weight_init: Literal["normal", "uniform"] = "uniform"

    enforce_max_neighbors_strictly: bool = True
    avg_num_nodes: float = 77.81317
    avg_degree: float = 23.395238876342773

    use_energy_lin_ref: bool = False
    load_energy_lin_ref: bool = False

    activation_checkpoint: bool = False
    regress_forces: bool = True
    regress_energy: bool = True
    direct_forces: bool = False

    max_neighbors: int = 20
    max_radius: float = 12.0
    max_num_elements: int = 90

    use_pbc: bool = True
    use_pbc_single: bool = False
    otf_graph: bool = True

    mof_feature_fusion: dict = {}
    
    @classmethod
    def base(cls):
        return cls(
            num_targets=1,
            num_layers=8,
            sphere_channels=128,
            attn_hidden_channels=64,
            num_heads=8,
            attn_alpha_channels=64,
            attn_value_channels=16,
            ffn_hidden_channels=128,
            norm_type="layer_norm_sh",
            lmax_list=[4],
            mmax_list=[2],
            grid_resolution=18,
            num_sphere_samples=128,
            edge_channels=128,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=512,
            attn_activation="silu",
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation="silu",
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,
            alpha_drop=0.1,
            drop_path_rate=0.1,
            proj_drop=0.0,
            weight_init="uniform",
            enforce_max_neighbors_strictly=True,
            avg_num_nodes=77.81317,
            avg_degree=23.395238876342773,
            use_energy_lin_ref=False,
            load_energy_lin_ref=False,
            activation_checkpoint=False,
            regress_forces=True,
            regress_energy=True,
            direct_forces=False,
            max_neighbors=20,
            max_radius=12.0,
            max_num_elements=90,
            use_pbc=True,
            use_pbc_single=False,
            otf_graph=True,
        )

    @classmethod
    def very_small_qm9(cls):
        return cls(
            num_targets=1,
            num_layers=5,
            sphere_channels=96,
            attn_hidden_channels=48,
            num_heads=4,
            attn_alpha_channels=64,
            attn_value_channels=16,
            ffn_hidden_channels=96,
            norm_type="layer_norm_sh",
            lmax_list=[4],
            mmax_list=[4],
            grid_resolution=18,
            num_sphere_samples=128,
            edge_channels=96,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=128,
            attn_activation="silu",
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation="silu",
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,
            alpha_drop=0.1,
            drop_path_rate=0.1,
            proj_drop=0.0,
            weight_init="uniform",
            enforce_max_neighbors_strictly=True,
            avg_num_nodes=77.81317,
            avg_degree=23.395238876342773,
            use_energy_lin_ref=False,
            load_energy_lin_ref=False,
            activation_checkpoint=False,
            regress_forces=True,
            regress_energy=True,
            direct_forces=False,
            max_neighbors=500,
            max_radius=5.0,
            max_num_elements=90,
            use_pbc=True,
            use_pbc_single=False,
            otf_graph=True,
        )

    @classmethod
    def small(cls):
        return cls(
            num_targets=1,
            num_layers=8,
            sphere_channels=128,
            attn_hidden_channels=64,
            num_heads=8,
            attn_alpha_channels=64,
            attn_value_channels=16,
            ffn_hidden_channels=128,
            norm_type="layer_norm_sh",
            lmax_list=[4],
            mmax_list=[2],
            grid_resolution=18,
            num_sphere_samples=128,
            edge_channels=128, 
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=512,
            attn_activation="silu",
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation="silu",
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,
            alpha_drop=0.1,
            drop_path_rate=0.1,
            proj_drop=0.0,
            weight_init="uniform",
            enforce_max_neighbors_strictly=True,
            avg_num_nodes=77.81317,
            avg_degree=23.395238876342773,
            use_energy_lin_ref=False,
            load_energy_lin_ref=False,
            activation_checkpoint=False,
            regress_forces=True,
            regress_energy=True,
            direct_forces=False,
            max_neighbors=20,
            max_radius=12.0,
            max_num_elements=90,
            use_pbc=True,
            use_pbc_single=False,
            otf_graph=True,
        )


    @classmethod
    def medium(cls):
        return cls(
            num_targets=1,
            num_layers=12,
            sphere_channels=128,
            attn_hidden_channels=64,
            num_heads=8,
            attn_alpha_channels=64,
            attn_value_channels=16,
            ffn_hidden_channels=128,
            norm_type="layer_norm_sh",
            lmax_list=[6],
            mmax_list=[2],
            grid_resolution=18,
            num_sphere_samples=128,
            edge_channels=128,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=512,
            attn_activation="silu",
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation="silu",
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,
            alpha_drop=0.1,
            drop_path_rate=0.05,
            proj_drop=0.0,
            weight_init="uniform",
            enforce_max_neighbors_strictly=True,
            avg_num_nodes=77.81317,
            avg_degree=23.395238876342773,
            use_energy_lin_ref=False,
            load_energy_lin_ref=False,
            activation_checkpoint=False,
            regress_forces=True,
            regress_energy=True,
            direct_forces=False,
            max_neighbors=20,
            max_radius=12.0,
            max_num_elements=90,
            use_pbc=True,
            use_pbc_single=False,
            otf_graph=True,
        )


    @classmethod
    def large(cls):
        return cls(
            num_targets=1,
            num_layers=20,
            sphere_channels=128,
            attn_hidden_channels=64,
            num_heads=8,
            attn_alpha_channels=64,
            attn_value_channels=16,
            ffn_hidden_channels=128,
            norm_type="layer_norm_sh",
            lmax_list=[6],
            mmax_list=[3],
            grid_resolution=18,
            num_sphere_samples=128,
            edge_channels=128,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=512,
            attn_activation="silu",
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation="silu",
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,
            alpha_drop=0.1,
            drop_path_rate=0.1,
            proj_drop=0.0,
            weight_init="uniform",
            enforce_max_neighbors_strictly=True,
            avg_num_nodes=77.81317,
            avg_degree=23.395238876342773,
            use_energy_lin_ref=False,
            load_energy_lin_ref=False,
            activation_checkpoint=False,
            regress_forces=True,
            regress_energy=True,
            direct_forces=False,
            max_neighbors=20,
            max_radius=12.0,
            max_num_elements=90,
            use_pbc=True,
            use_pbc_single=False,
            otf_graph=True,
        )
    
    @classmethod
    def small_odac(cls):
        return cls(
            num_targets=1,
            num_layers=8,
            sphere_channels=128,
            attn_hidden_channels=64,
            num_heads=8,
            attn_alpha_channels=64,
            attn_value_channels=16,
            ffn_hidden_channels=128,
            norm_type="layer_norm_sh",
            lmax_list=[4],
            mmax_list=[2],
            grid_resolution=18,
            num_sphere_samples=128,
            edge_channels=128,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=512,
            attn_activation="silu",
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation="silu",
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,
            alpha_drop=0.1,
            drop_path_rate=0.1,
            proj_drop=0.0,
            weight_init="uniform",
            enforce_max_neighbors_strictly=True,
            avg_num_nodes=192.561,
            avg_degree=21.024127419363214,
            use_energy_lin_ref=False,
            load_energy_lin_ref=False,
            activation_checkpoint=False,
            regress_forces=True,
            regress_energy=True,
            direct_forces=False,
            max_neighbors=20,
            max_radius=8.0,
            max_num_elements=100,
            use_pbc=True,
            use_pbc_single=False,
            otf_graph=True,
        )
    
    @classmethod
    def large_odac(cls):
        return cls(
            num_targets=1,
            num_layers=20,
            sphere_channels=128,
            attn_hidden_channels=64,
            num_heads=8,
            attn_alpha_channels=64,
            attn_value_channels=16,
            ffn_hidden_channels=128,
            norm_type="layer_norm_sh",
            lmax_list=[6],
            mmax_list=[3],
            grid_resolution=18,
            num_sphere_samples=128,
            edge_channels=128,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=512,
            attn_activation="silu",
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation="silu",
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,
            alpha_drop=0.1,
            drop_path_rate=0.1,
            proj_drop=0.0,
            weight_init="uniform",
            enforce_max_neighbors_strictly=True,
            avg_num_nodes=192.561,
            avg_degree=21.024127419363214,
            use_energy_lin_ref=False,
            load_energy_lin_ref=False,
            activation_checkpoint=False,
            regress_forces=True,
            regress_energy=True,
            direct_forces=False,
            max_neighbors=20,
            max_radius=8.0,
            max_num_elements=100,
            use_pbc=True,
            use_pbc_single=False,
            otf_graph=True,
        )



class AutoregressiveLlamaEquiformerV2Config(TypedConfig):
    name: Literal["AutoregLlamaEquiformerV2", "autoreg_llama_equiformer_v2"] = "autoreg_llama_equiformer_v2"  

    num_targets: int = 1
    num_layers: int = 8
    sphere_channels: int = 128
    attn_hidden_channels: int = 64
    num_heads: int = 8
    attn_alpha_channels: int = 64
    attn_value_channels: int = 16
    ffn_hidden_channels: int = 128
    norm_type: str = "rms_norm_sh"

    lmax_list: list[int] = [4]
    mmax_list: list[int] = [2]
    grid_resolution: int = 18

    num_sphere_samples: int = 128

    edge_channels: int = 128
    use_atom_edge_embedding: bool = True
    share_atom_edge_embedding: bool = False
    use_m_share_rad: bool = False
    distance_function: Literal["gaussian"] = "gaussian"
    num_distance_basis: int = 512

    attn_activation: str = "scaled_silu"
    use_s2_act_attn: bool = False
    use_attn_renorm: bool = True
    ffn_activation: str = "scaled_silu"
    use_gate_act: bool = False
    use_grid_mlp: bool = True
    use_sep_s2_act: bool = True

    alpha_drop: float = 0.1
    drop_path_rate: float = 0.1
    proj_drop: float = 0.0
    weight_init: Literal["normal", "uniform"] = "normal"

    # enforce_max_neighbors_strictly: bool = True
    avg_num_nodes: float = 77.81317
    avg_degree: float = 23.395238876342773

    use_energy_lin_ref: bool = False
    load_energy_lin_ref: bool = False

    activation_checkpoint: bool = False
    regress_forces: bool = True
    regress_energy: bool = True
    direct_forces: bool = False

    max_neighbors: int = 20
    max_radius: float = 12.0
    max_num_elements: int = 90

    use_pbc: bool = True
    use_pbc_single: bool = False
    otf_graph: bool = True


    llm_checkpoint: str = "HuggingFaceTB/SmolLM2-1.7B"
    use_position_enc: bool = False
    add_prop_pred: bool= False
    max_token: int | None = 512
    init_rand_special_tokens: bool = True   

    @classmethod
    def base(cls):
        return cls(
            norm_type="layer_norm_sh", # to match equiformer_v2_N@8_L@4_M@2_debug_only_autoreg_LLMS03.yml
        )

