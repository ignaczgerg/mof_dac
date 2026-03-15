# from dataclasses import dataclass
from pydantic.dataclasses import dataclass


from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch
from jmp.lightning import TypedConfig
from typing_extensions import override
from .custom_types import GraphAttentionData

class EScAIPBackboneOutput(TypedDict):
    data: GraphAttentionData | None
    node_features: Any 
    edge_features: Any

@dataclass
class GlobalConfigs:
    regress_forces: bool
    direct_force: bool
    hidden_size: int  # divisible by 2 and num_heads
    batch_size: int
    activation: Literal[
        "squared_relu", "gelu", "leaky_relu", "relu", "smelu", "star_relu"
    ]
    use_compile: bool = True
    use_padding: bool = True


@dataclass
class MolecularGraphConfigs:
    use_pbc: bool
    use_pbc_single: bool
    otf_graph: bool
    max_neighbors: int
    max_radius: float
    max_num_elements: int
    max_num_nodes_per_batch: int
    enforce_max_neighbors_strictly: bool
    distance_function: Literal["gaussian", "sigmoid", "linearsigmoid", "silu"]


@dataclass
class GraphNeuralNetworksConfigs:
    num_layers: int
    atom_embedding_size: int
    node_direction_embedding_size: int
    node_direction_expansion_size: int
    edge_distance_expansion_size: int
    edge_distance_embedding_size: int
    atten_name: Literal[
        "math",
        "memory_efficient",
        "flash",
    ]
    atten_num_heads: int
    readout_hidden_layer_multiplier: int
    output_hidden_layer_multiplier: int
    ffn_hidden_layer_multiplier: int
    use_angle_embedding: bool = True
    energy_reduce: Literal["sum", "mean"] = "mean"


@dataclass
class RegularizationConfigs:
    mlp_dropout: float
    atten_dropout: float
    stochastic_depth_prob: float
    normalization: Literal["layernorm", "rmsnorm", "skip"]

class EScAIPBackboneConfig(TypedConfig):
    name: Literal["EScAIP", "escaip"] = "EScAIP"  
    
    # Global Configs
    regress_forces: bool = True
    regress_energy: bool = True
    direct_forces: bool = True
    hidden_size: int = 512  # divisible by 2 and num_heads
    batch_size: int = 8
    activation: Literal[
        "squared_relu", "gelu", "leaky_relu", "relu", "smelu", "star_relu"
    ] = "gelu"
    use_compile: bool = True
    use_padding: bool = True
    
    # Molecular Graph Configs
    use_pbc: bool = True
    use_pbc_single: bool = False
    otf_graph: bool = True
    max_neighbors: int = 10 # 20
    max_radius: float = 12.0
    max_num_elements: int = 120 # 90 for small
    max_num_nodes_per_batch: int = 540 # 150 for small 
    enforce_max_neighbors_strictly: bool = True 
    distance_function: Literal["gaussian", "sigmoid", "linearsigmoid", "silu"] = "gaussian"
    
    # Graph Neural Networks Configs
    num_layers: int = 10
    atom_embedding_size: int = 128
    node_direction_embedding_size: int = 64
    node_direction_expansion_size: int = 10
    edge_distance_expansion_size: int = 600
    edge_distance_embedding_size: int = 512
    atten_name: Literal[
        "math",
        "memory_efficient",
        "flash",
    ] = "memory_efficient"
    atten_num_heads: int = 8
    readout_hidden_layer_multiplier: int = 2
    output_hidden_layer_multiplier: int = 2
    ffn_hidden_layer_multiplier: int = 2
    use_angle_embedding: bool = True
    energy_reduce: Literal["sum", "mean"] = "mean"
    
    # Regularization Configs
    mlp_dropout: float = 0.1
    atten_dropout: float = 0.1
    stochastic_depth_prob: float = 0.0
    normalization: Literal["layernorm", "rmsnorm", "skip"] = "rmsnorm"
    
    @property
    def emb_size_atom(self):
        return self.atom_embedding_size
    
    @property
    def num_elements(self):
        return self.max_num_elements
    

    @property
    def global_cfg(self):
        return GlobalConfigs(
            regress_forces=self.regress_forces,
            direct_force=self.direct_forces,
            hidden_size=self.hidden_size,
            batch_size=self.batch_size,
            activation=self.activation,
            use_compile=self.use_compile,
            use_padding=self.use_padding,
        )

    @property
    def molecular_graph_cfg(self):
        return MolecularGraphConfigs(
            use_pbc=self.use_pbc,
            use_pbc_single=self.use_pbc_single,
            otf_graph=self.otf_graph,
            max_neighbors=self.max_neighbors,
            max_radius=self.max_radius,
            max_num_elements=self.max_num_elements,
            max_num_nodes_per_batch=self.max_num_nodes_per_batch,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            distance_function=self.distance_function,
        )
        
    @property
    def gnn_cfg(self):
        return GraphNeuralNetworksConfigs(
            num_layers=self.num_layers,
            atom_embedding_size=self.atom_embedding_size,
            node_direction_embedding_size=self.node_direction_embedding_size,
            node_direction_expansion_size=self.node_direction_expansion_size,
            edge_distance_expansion_size=self.edge_distance_expansion_size,
            edge_distance_embedding_size=self.edge_distance_embedding_size,
            atten_name=self.atten_name,
            atten_num_heads=self.atten_num_heads,
            readout_hidden_layer_multiplier=self.readout_hidden_layer_multiplier,
            output_hidden_layer_multiplier=self.output_hidden_layer_multiplier,
            ffn_hidden_layer_multiplier=self.ffn_hidden_layer_multiplier,
            use_angle_embedding=self.use_angle_embedding,
            energy_reduce=self.energy_reduce,
        )
    
    @property
    def reg_cfg(self):
        return RegularizationConfigs(
            mlp_dropout=self.mlp_dropout,
            atten_dropout=self.atten_dropout,
            stochastic_depth_prob=self.stochastic_depth_prob,
            normalization=self.normalization,
        )
        
        
    @classmethod
    def base(cls):
        return cls()
    
    @classmethod
    def small(cls):
        return cls(
            num_layers=8, # to match small EqV2
        )
    
    @classmethod
    def very_small(cls):
        return cls(
            regress_forces=True,
            regress_energy=True,
            direct_forces=True,
            hidden_size=512,  # divisible by 2 and num_heads
            batch_size=12,
            activation="gelu",
            use_compile=True,
            use_padding=True,
            
            # Molecular Graph Configs
            use_pbc=True,
            use_pbc_single=False,
            otf_graph=True,
            max_neighbors=10,  # 20
            max_radius=12.0,
            max_num_elements=90,  # 90 for small
            max_num_nodes_per_batch=150,  # 150 for small 
            enforce_max_neighbors_strictly=True,
            distance_function="gaussian",
            
            # Graph Neural Networks Configs
            num_layers=10,
            atom_embedding_size=128,
            node_direction_embedding_size=64,
            node_direction_expansion_size=10,
            edge_distance_expansion_size=600,
            edge_distance_embedding_size=512,
            atten_name="memory_efficient",
            atten_num_heads=8,
            readout_hidden_layer_multiplier=2,
            output_hidden_layer_multiplier=2,
            ffn_hidden_layer_multiplier=2,
            use_angle_embedding=True,
            energy_reduce="mean",
            
            # Regularization Configs
            mlp_dropout=0.05,
            atten_dropout=0.1,
            stochastic_depth_prob=0.0,
            normalization="rmsnorm",
        )
    
    @classmethod
    def medium(cls):
        raise NotImplementedError("Medium backbone not implemented yet")
    
    @classmethod
    def large(cls):
        raise NotImplementedError("Large backbone not implemented yet")



@dataclass
class EScAIPConfigs:
    global_cfg: GlobalConfigs
    molecular_graph_cfg: MolecularGraphConfigs
    gnn_cfg: GraphNeuralNetworksConfigs
    reg_cfg: RegularizationConfigs


# def init_configs(cls: type[EScAIPConfigs], kwargs: dict[str, Any]) -> EScAIPConfigs:
#     """
#     Initialize a dataclass with the given kwargs.
#     """
#     init_kwargs = {}
#     for field in fields(cls):
#         if is_dataclass(field.type):
#             init_kwargs[field.name] = init_configs(field.type, kwargs)
#         elif field.name in kwargs:
#             init_kwargs[field.name] = kwargs[field.name]
#         elif field.default is not None:
#             init_kwargs[field.name] = field.default
#         else:
#             raise ValueError(
#                 f"Missing required configuration parameter: '{field.name}'"
#             )

#     return cls(**init_kwargs)
