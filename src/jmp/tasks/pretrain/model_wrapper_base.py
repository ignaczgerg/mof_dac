"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations
import fnmatch
import itertools
import math
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeAlias, assert_never, cast
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from jmp.lightning import Base, BaseConfig, Field, LightningModuleBase, TypedConfig

from jmp.lightning.util.typed import TypedModuleDict, TypedModuleList
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from torch_scatter import scatter
from typing_extensions import TypedDict, TypeVar, override

from ...models.gemnet.backbone import GemNetOCBackbone, GOCBackboneOutput
from ...models.gemnet.config import GOCBackboneConfig
from ...models.schnet.backbone import SchNetBackbone, SchNetBackboneOutput
from ...models.schnet.config import SchNetBackboneConfig
from ...models.EScAIP.EScAIP import EScAIPBackbone, EScAIPEnergyHead
from ...models.EScAIP.config import EScAIPBackboneConfig, EScAIPBackboneOutput
from ...models.equiformer_v2.equiformer_v2 import (
    EquiformerV2Backbone, AutoregressiveLlamaEquiformerV2Backbone,
    AutoregressiveEquiformerV2Backbone,
    EquiformerV2EnergyHead, EquiformerV2ForceHead, EqV2NodeScalarHead,
    EqV2ScalarHead, EqV2ClassificationHead, EqV2VectorHead
)
from ...models.equiformer_v2.config import (
    EquiformerV2Config, EquiformerV2BackboneOutput, AutoregressiveLlamaEquiformerV2Config
)


from ...utils.state_dict import load_state_dict

log = getLogger(__name__)


BackboneConfig: TypeAlias = Annotated[
    GOCBackboneConfig | SchNetBackboneConfig | EScAIPBackboneConfig | EquiformerV2Config |
    AutoregressiveLlamaEquiformerV2Config,
    Field(discriminator="name"),
]

Backbone: TypeAlias = (GemNetOCBackbone | SchNetBackbone | EScAIPBackbone | 
                        EquiformerV2Backbone | AutoregressiveLlamaEquiformerV2Backbone)
BackboneOutput: TypeAlias = (GOCBackboneOutput | SchNetBackboneOutput | 
                             EScAIPBackboneOutput | EquiformerV2BackboneOutput)



class BinaryClassificationTargetConfig(TypedConfig):
    name: str
    """The name of the target"""
    num_classes: int
    """The number of classes for the target"""

    pos_weight: float | None = None
    """The positive weight for the target"""

    @override
    def __post_init__(self):
        super().__post_init__()

        if self.num_classes != 2:
            raise ValueError(
                f"Binary classification target {self.name} has {self.num_classes} classes"
            )


class MulticlassClassificationTargetConfig(TypedConfig):
    name: str
    """The name of the target"""
    num_classes: int
    """The number of classes for the target"""

    class_weights: list[float] | None = None
    """The class weights for the target"""
    dropout: float | None = None
    """The dropout probability to use before the output layer"""


_TConfig = TypeVar("_TConfig", bound=BaseConfig)
class ModelWrapperBase(Base[_TConfig], nn.Module,  Generic[_TConfig]):
    # def forward(self, data: BaseData) -> "OutputHeadInput": ...
    def validate_config(self, config: _TConfig) -> None: ...

TModel = TypeVar('TModel', bound=ModelWrapperBase)

TConfig = TypeVar("TConfig", bound="PretrainConfig")




class GemNetModelWrapper(ModelWrapperBase[TConfig], Base[TConfig]):
    BackboneConfigType: TypeAlias = GOCBackboneConfig
    BackboneType: TypeAlias = GemNetOCBackbone
    BackboneOutputType: TypeAlias = GOCBackboneOutput
    
    class Embedding(Base[TConfig], nn.Module):
        @override
        def __init__(self, hparams: TConfig):
            super().__init__(hparams)

            self.atom_embedding = nn.Embedding(
                num_embeddings=self.config.embedding.num_elements,
                embedding_dim=self.config.embedding.embedding_size,
            )

        @override
        def forward(self, data: BaseData):
            atomic_numbers = data.atomic_numbers - 1
            x = self.atom_embedding(atomic_numbers)
            return x
        
        
    class PretrainOutputHead(Base[TConfig], nn.Module):
        @override
        def __init__(self, config: TConfig, backbone: GemNetOCBackbone):
            super().__init__(config)

            def dims(
                emb_size: int,
                *,
                num_targets: int = self.config.backbone.num_targets,
                num_mlps: int = self.config.output.num_mlps,
            ):
                return ([emb_size] * num_mlps) + [num_targets]

            self.out_energy = TypedModuleList(
                [
                    self.mlp(
                        dims(self.config.backbone.emb_size_atom),
                        activation=self.config.activation_cls,
                    )
                    for _ in self.config.tasks
                ]
            )
            self.out_forces = TypedModuleList(
                [
                    self.mlp(
                        dims(self.config.backbone.emb_size_edge),
                        activation=self.config.activation_cls,
                    )
                    for _ in self.config.tasks
                ]
            )

        @override
        def forward(self, data: BaseData, backbone_out: GOCBackboneOutput):
            energy = backbone_out["energy"]
            forces = backbone_out["forces"]
            V_st = backbone_out["V_st"]
            idx_t = backbone_out["idx_t"]

            batch: torch.Tensor = data.batch
            n_molecules = int(torch.max(batch).item() + 1)
            n_atoms = data.atomic_numbers.shape[0]

            energy_list: list[torch.Tensor] = []
            forces_list: list[torch.Tensor] = []

            for energy_mlp, forces_mlp, task in zip(
                self.out_energy, self.out_forces, self.config.tasks
            ):
                E_t = energy_mlp(energy)  # (n_atoms, 1)
                E_t = scatter(
                    E_t,
                    batch,
                    dim=0,
                    dim_size=n_molecules,
                    reduce=task.node_energy_reduction,
                )
                energy_list.append(E_t)  # (bsz, 1)

                F_st = forces_mlp(forces)  # (n_edges, 1)
                F_st = F_st * V_st  # (n_edges, 3)
                F_t = scatter(F_st, idx_t, dim=0, dim_size=n_atoms, reduce="sum")
                forces_list.append(F_t)  # (n_atoms, 3)

            E, _ = pack(energy_list, "bsz *")
            F, _ = pack(forces_list, "n_atoms p *")

            return E, F
    
    
    def __init__(self, config: TConfig):
        super().__init__(config)     

    def _model_validate_config(self, config) -> None:
        assert (
            config.activation.lower() == config.backbone.activation.lower()
        ), f"{config.activation=} != {config.backbone.activation=}"

        assert (
            config.embedding.num_elements == config.backbone.num_elements
        ), f"{config.embedding.num_elements=} != {config.backbone.num_elements=}"
        assert (
            config.embedding.embedding_size == config.backbone.emb_size_atom
        ), f"{config.embedding.embedding_size=} != {config.backbone.emb_size_atom=}"

    @override
    def validate_config(self, config: TConfig) -> None:
        self._model_validate_config(config)
        
   


class SchNetModelWrapper(ModelWrapperBase[TConfig]):
    BackboneConfigType: TypeAlias = SchNetBackboneConfig
    BackboneType: TypeAlias = SchNetBackbone
    BackboneOutputType: TypeAlias = SchNetBackboneOutput
   

            
    def __init__(self, config: TConfig):
        super().__init__(config)

    
        


class EScAIPModelWrapper(ModelWrapperBase[TConfig]):
    BackboneConfigType: TypeAlias = EScAIPBackboneConfig
    BackboneType: TypeAlias = EScAIPBackbone
    BackboneOutputType: TypeAlias = EScAIPBackboneOutput
   

    def __init__(self, config: TConfig):
        super().__init__(config)
        


class EquiformerV2ModelWrapper(ModelWrapperBase[TConfig]):
    BackboneConfigType: TypeAlias = EquiformerV2Config
    BackboneType: TypeAlias = EquiformerV2Backbone
    BackboneOutputType: TypeAlias = EquiformerV2BackboneOutput
   
 

    class PretrainOutputHead(Base[TConfig], nn.Module):
        def __init__(self, config: TConfig, backbone: EquiformerV2Backbone):
            super().__init__(config)
            self.tasks = config.tasks  # List of tasks

            # Create multiple heads, one for each task
            # self.energy_heads = nn.ModuleList(
            #     [EquiformerV2EnergyHead(backbone) for _ in self.tasks]
            # )
            # self.force_heads = nn.ModuleList(
            #     [EquiformerV2ForceHead(backbone) for _ in self.tasks]
            # )
            
            # Single shared output head
            self.energy_head = EquiformerV2EnergyHead(backbone)
            self.force_head = EquiformerV2ForceHead(backbone)

        def forward(self, batch, backbone_out):
            """
            Args:
                batch: Input batch containing atomic and molecular data.
                backbone_out: Output from the EquiformerV2 backbone.
            Returns:
                energy_list: List of energy predictions for each task.
                forces_list: List of force predictions for each task.
            """
            energy_list = []
            forces_list = []
            
            # feature_dict = {}
            # feature_dict = backbone_out['feature_dict']
            # out_embedding = backbone_out["node_embedding"].embedding
            # node_size = out_embedding.shape[0]
            # feature_dict['node'] = out_embedding.reshape(node_size, -1)

            # Multiple output heads for each task
            # for energy_head, force_head, task in zip(
            #     self.energy_heads, self.force_heads, self.tasks
            # ):
            #     energy = energy_head(batch, backbone_out)["energy"]
            #     forces = force_head(batch, backbone_out)["forces"]
            #     energy_list.append(energy)
            #     forces_list.append(forces)

            # energy_tensor = torch.stack(energy_list, dim=-1)
            # forces_tensor = torch.stack(forces_list, dim=-1)

            # return energy_tensor, forces_tensor
            
            return (
                torch.stack([self.energy_head(batch, backbone_out)["energy"] for _ in self.tasks], dim=-1),
                torch.stack([self.force_head(batch, backbone_out)["forces"] for _ in self.tasks], dim=-1)
            )
        

class AutoregressiveEquiformerV2ModelWrapper(ModelWrapperBase[TConfig]):
    BackboneConfigType: TypeAlias = EquiformerV2Config
    BackboneType: TypeAlias = AutoregressiveEquiformerV2Backbone
    BackboneOutputType: TypeAlias = EquiformerV2BackboneOutput
   
 
    # For this class, we moved the heads (positions, atomic number) to the backbone initialization for reproducibility reasons with the orig equiformer_v2 repo. 
    class PretrainOutputHead(Base[TConfig], nn.Module):
        def __init__(self, config: TConfig, backbone: AutoregressiveEquiformerV2Backbone):
            super().__init__(config)
            self.tasks = config.tasks  # List of tasks
            self.multiple_heads = config.multi_heads
 
            if self.multiple_heads:
                # Create multiple heads, one for each task
                # self.energy_heads = nn.ModuleList(
                #     [EquiformerV2EnergyHead(backbone) for _ in self.tasks]
                # )
                self.coord_head = nn.ModuleList(
                    [EqV2VectorHead(backbone, output_name="positions") for _ in self.tasks]
                )
                # self.coord_head = nn.ModuleList(
                #     [EqV2NodeScalarHead(
                #         backbone, output_channels=3, output_name="positions") for _ in self.tasks]
                # )
                self.atomic_num_head = nn.ModuleList( 
                    [EqV2NodeScalarHead(
                        backbone, output_channels=backbone.max_num_elements, 
                        output_name="atomic_numbers") for _ in self.tasks]
                )
            
            else:
                # Single shared output head
                # self.energy_head = EquiformerV2EnergyHead(backbone)
                self.coord_head = EqV2VectorHead(backbone, output_name="positions")
                # self.coord_head = EqV2NodeScalarHead(
                #     backbone, output_channels=3, output_name="positions")
                self.atomic_num_head = EqV2NodeScalarHead(
                    backbone, output_channels=backbone.max_num_elements, 
                    output_name="atomic_numbers"
                )

            

        def forward(self, batch, backbone_out):
            """
            Args:
                batch: Input batch containing atomic and molecular data.
                backbone_out: Output from the EquiformerV2 backbone.
            Returns:
                energy_list: List of energy predictions for each task.
                forces_list: List of force predictions for each task.
                positions_list: List of positions predictions for each task.
                atomic_numbers_list: List of atomic numbers predictions for each task.
            """
            # energy_list = []
            # forces_list = []
            positions_list = []
            atomic_numbers_list = []
            
            # feature_dict = {}
            # feature_dict = backbone_out['feature_dict']
            # out_embedding = backbone_out["node_embedding"].embedding
            # node_size = out_embedding.shape[0]
            # feature_dict['node'] = out_embedding.reshape(node_size, -1)
            if self.multiple_heads:
                for coord_head, atomic_num_head, _ in zip(
                    self.coord_head, self.atomic_num_head, self.tasks
                ):
                    positions = coord_head(batch, backbone_out)["positions"]
                    # atomic_numbers = F.softmax(
                    #     atomic_num_head(batch, backbone_out)["atomic_numbers"], dim=-1
                    #     ).argmax(dim=-1)
                    atomic_numbers = atomic_num_head(batch, backbone_out)["atomic_numbers"]
                    positions_list.append(positions)
                    atomic_numbers_list.append(atomic_numbers)
                
                # Mulitple output heads for each task
                return (
                    None, # energy_list
                    None, # forces_list
                    torch.stack(positions_list, dim=-1),
                    torch.stack(atomic_numbers_list, dim=-1),
                )
            
            else:
                # Single shared output head
                positions = self.coord_head(batch, backbone_out)["positions"]
                atomic_numbers = self.atomic_num_head(batch, backbone_out)["atomic_numbers"]
                
                return (
                    None, # energy_list
                    None, # forces_list
                    torch.stack([positions for _ in self.tasks], dim=-1),
                    torch.stack([atomic_numbers for _ in self.tasks], dim=-1)
                )            
            # Single shared output head (defined in backbone)
            # return (
            #     None, # energy_list
            #     None, # forces_list
            #     torch.stack([backbone_out["nodes_positions"] for _ in self.tasks], dim=-1),
            #     torch.stack([backbone_out["nodes_atom_num"] for _ in self.tasks], dim=-1)
            # )
            
            





class AutoregLlamaEquiformerV2ModelWrapper(ModelWrapperBase[TConfig]):
    BackboneConfigType: TypeAlias = AutoregressiveLlamaEquiformerV2Config
    BackboneType: TypeAlias = AutoregressiveLlamaEquiformerV2Backbone
    BackboneOutputType: TypeAlias = EquiformerV2BackboneOutput
   


    def __init__(self, config: TConfig):
        super().__init__(config)




def __test(_int: int):
    pass