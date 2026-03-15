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
from einops import rearrange
from jmp.lightning import Base, BaseConfig, Field, LightningModuleBase, TypedConfig

from jmp.lightning.util.typed import TypedModuleDict
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from torch_scatter import scatter
from typing_extensions import TypedDict, TypeVar, override

from ...modules.transforms.normalize import NormalizationConfig

from ...models.gemnet.backbone import GemNetOCBackbone, GOCBackboneOutput
from ...models.gemnet.config import GOCBackboneConfig
from ...models.schnet.backbone import SchNetBackbone, SchNetBackboneOutput
from ...models.schnet.config import SchNetBackboneConfig
from ...models.EScAIP.EScAIP import EScAIPBackbone, EScAIPEnergyHead
from ...models.EScAIP.config import EScAIPBackboneConfig, EScAIPBackboneOutput
from ...models.equiformer_v2.equiformer_v2 import (
    EquiformerV2Backbone, AutoregressiveLlamaEquiformerV2Backbone, 
    EqV2ScalarHead, EqV2ClassificationHead, ScalarRBFEncoder
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
    def forward(self, data: BaseData) -> "OutputHeadInput": ...
    def validate_config(self, config: _TConfig) -> None: ...

TModel = TypeVar('TModel', bound=ModelWrapperBase)

TConfig = TypeVar("TConfig", bound="FinetuneConfigBase")


class OutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: BackboneOutput


class GraphBinaryClassificationOutputHead(
    Base[TConfig], nn.Module, Generic[TConfig]):
    @override
    def __init__(
        self,
        config: TConfig,
        classification_config: BinaryClassificationTargetConfig,
        reduction: str | None = None,
    ):
        super().__init__(config)

        assert (
            classification_config.num_classes == 2
        ), "Only binary classification supported"

        if reduction is None:
            reduction = self.config.graph_scalar_reduction_default

        self.out_mlp = self.mlp(
            ([self.config.backbone.emb_size_atom] * self.config.output.num_mlps) + [1],
            activation=self.config.activation_cls,
        )
        self.classification_config = classification_config
        self.reduction = reduction

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_molecules = int(torch.max(data.batch).item() + 1)

        output = self.out_mlp(backbone_output["energy"])  # (n, num_classes)
        output = scatter(
            output,
            data.batch,
            dim=0,
            dim_size=n_molecules,
            reduce=self.reduction,
        )  # (bsz, num_classes)
        output = rearrange(output, "b 1 -> b")
        return output


class NodeVectorOutputHead(Base[TConfig], nn.Module, Generic[TConfig]):
    @override
    def __init__(
        self,
        config: TConfig,
        reduction: str | None = None,
    ):
        super().__init__(config)

        if reduction is None:
            reduction = self.config.graph_scalar_reduction_default

        self.out_mlp = self.mlp(
            ([self.config.backbone.emb_size_edge] * self.config.output.num_mlps)
            + [self.config.backbone.num_targets],
            activation=self.config.activation_cls,
        )
        self.reduction = reduction

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_atoms = data.atomic_numbers.shape[0]

        output = self.out_mlp(backbone_output["forces"])
        output = output * backbone_output["V_st"]  # (n_edges, 3)
        output = scatter(
            output,
            backbone_output["idx_t"],
            dim=0,
            dim_size=n_atoms,
            reduce=self.reduction,
        )
        return output




class GemNetModelWrapper(ModelWrapperBase[TConfig], Base[TConfig]):
    BackboneConfigType: TypeAlias = GOCBackboneConfig
    BackboneType: TypeAlias = GemNetOCBackbone
    BackboneOutputType: TypeAlias = GOCBackboneOutput
    
    class GraphScalarOutputHead(Base[TConfig], nn.Module): # type: ignore
        @override
        def __init__(
            self,
            config: TConfig,
            backbone: GemNetOCBackbone,
            reduction: str | None = None,
        ):
            super().__init__(config)

        
            if reduction is None:
                reduction = self.config.graph_scalar_reduction_default

            self.out_mlp = self.mlp(
                ([self.config.backbone.emb_size_atom] * self.config.output.num_mlps)
                + [self.config.backbone.num_targets],
                activation=self.config.activation_cls,
            )
            
            self.reduction = reduction

        @override
        def forward(
            self,
            input: OutputHeadInput,
            *,
            scale: torch.Tensor | None = None,
            shift: torch.Tensor | None = None,
        ) -> torch.Tensor:
            data = input["data"]
            backbone_output = input["backbone_output"]

            
            n_molecules = int(torch.max(data.batch).item() + 1)

            output = self.out_mlp(backbone_output["energy"])  # (n_atoms, 1) # type: ignore
            if scale is not None:
                output = output * scale
            if shift is not None:
                output = output + shift

            output = scatter(
                output,
                data.batch,
                dim=0,
                dim_size=n_molecules,
                reduce=self.reduction, # type: ignore
            )  # (bsz, 1)
            output = rearrange(output, "b 1 -> b")
        
            return output
        
    class GraphMulticlassClassificationOutputHead(
        Base[TConfig], nn.Module, Generic[TConfig]
    ):
        @override
        def __init__(
            self,
            config: TConfig,
            classification_config: MulticlassClassificationTargetConfig,
            backbone: GemNetOCBackbone,
            reduction: str | None = None,
        ):
            super().__init__(config)

            if reduction is None:
                reduction = self.config.graph_scalar_reduction_default

            self.out_mlp = self.mlp(
                ([self.config.backbone.emb_size_atom] * self.config.output.num_mlps)
                + [classification_config.num_classes],
                activation=self.config.activation_cls,
            )
            self.classification_config = classification_config
            self.reduction = reduction

            self.dropout = None
            if classification_config.dropout:
                self.dropout = nn.Dropout(classification_config.dropout)

        @override
        def forward(self, input: OutputHeadInput) -> torch.Tensor:
            data = input["data"]
            n_molecules = int(torch.max(data.batch).item() + 1)

            x = input["backbone_output"]["energy"]
            if self.dropout is not None:
                x = self.dropout(x)

            x = self.out_mlp(x)  # (n, num_classes)
            x = scatter(
                x,
                data.batch,
                dim=0,
                dim_size=n_molecules,
                reduce=self.reduction,
            )  # (bsz, num_classes)
            return x
    
    def __init__(self, config: TConfig):
        super().__init__(config)     

    @override
    def validate_config(self, config) -> None:
            assert config.activation.lower() == config.backbone.activation.lower()
            assert config.embedding.num_elements == config.backbone.num_elements
            assert config.embedding.embedding_size == config.backbone.emb_size_atom
   
    def load_backbone_state_dict(
        self,
        _backbone: GemNetOCBackbone,
        _embedding: nn.Module,
        *,
        backbone: Mapping[str, Any],
        embedding: Mapping[str, Any],
        strict: bool = True,
    ):
        ignored_key_patterns = self.config.ckpt_load.ignored_key_patterns
        # If we're dumping the backbone's force out heads, then we need to ignore
        #   the unexpected keys for the force out MLPs and force out heads.
        if (
            not self.config.backbone.regress_forces
            or not self.config.backbone.direct_forces
        ):
            ignored_key_patterns.append("out_mlp_F.*")
            for block_idx in range(self.config.backbone.num_blocks + 1):
                ignored_key_patterns.append(f"out_blocks.{block_idx}.scale_rbf_F.*")
                ignored_key_patterns.append(f"out_blocks.{block_idx}.dense_rbf_F.*")
                ignored_key_patterns.append(f"out_blocks.{block_idx}.seq_forces.*")

        load_state_dict(
            _backbone,
            backbone,
            strict=strict,
            ignored_key_patterns=ignored_key_patterns,
            ignored_missing_keys=self.config.ckpt_load.ignored_missing_keys,
            ignored_unexpected_keys=self.config.ckpt_load.ignored_unexpected_keys,
        )
        if not self.config.ckpt_load.reset_embeddings:
            load_state_dict(_embedding, embedding, strict=strict)
        log.critical("Loaded backbone state dict (backbone and embedding).")

class SchNetModelWrapper(ModelWrapperBase[TConfig]):
    BackboneConfigType: TypeAlias = SchNetBackboneConfig
    BackboneType: TypeAlias = SchNetBackbone
    BackboneOutputType: TypeAlias = SchNetBackboneOutput
   
    class GraphScalarOutputHead(Base[TConfig], nn.Module): # type: ignore
        @override
        def __init__(
            self,
            config: TConfig,
            backbone: SchNetBackbone,
            reduction: str | None = None,
        ):
            super().__init__(config)

        
            if reduction is None:
                reduction = self.config.graph_scalar_reduction_default

            # In SchNet, out_mlp_F represents the scalar property
            self.out_mlp = nn.Sequential(nn.Identity())
            self.reduction = reduction

        @override
        def forward(
            self,
            input: OutputHeadInput,
            *,
            scale: torch.Tensor | None = None,
            shift: torch.Tensor | None = None,
        ) -> torch.Tensor:
            data = input["data"]
            backbone_output = input["backbone_output"]

            output = self.out_mlp(backbone_output["energy"])  # type: ignore
            output = rearrange(output, "b 1 -> b")
        
            return output
            
    def __init__(self, config: TConfig):
        super().__init__(config)

    
        

    def load_backbone_state_dict(
        self,
        *,
        backbone: Mapping[str, Any],
        embedding: Mapping[str, Any],
        strict: bool = True,
    ):
        raise NotImplementedError("SchNetModelWrapper does not implement backbone loading")
    

class EScAIPModelWrapper(ModelWrapperBase[TConfig]):
    BackboneConfigType: TypeAlias = EScAIPBackboneConfig
    BackboneType: TypeAlias = EScAIPBackbone
    BackboneOutputType: TypeAlias = EScAIPBackboneOutput
   
    class GraphScalarOutputHead(Base[TConfig], nn.Module): # type: ignore
        @override
        def __init__(
            self,
            config: TConfig,
            backbone: EScAIPBackbone,
            *,
            reduction: str | None = None,
        ):
            super().__init__(config)

        
            if reduction is None:
                reduction = self.config.graph_scalar_reduction_default

            self.out_mlp = EScAIPEnergyHead(backbone)
            self.reduction = reduction

        @override
        def forward(
            self,
            input: OutputHeadInput,
            *,
            scale: torch.Tensor | None = None,
            shift: torch.Tensor | None = None,
        ) -> torch.Tensor:
            data = input["data"]
            backbone_output = input["backbone_output"]

            output = self.out_mlp(data, backbone_output) 
            output = rearrange(output["energy"], "b 1 -> b")
            return output

    def __init__(self, config: TConfig):
        super().__init__(config)
        

        
    
    def load_backbone_state_dict(
        self,
        *,
        backbone: Mapping[str, Any],
        embedding: Mapping[str, Any],
        strict: bool = True,
    ):
        raise NotImplementedError("EScAIPModelWrapper does not implement backbone loading")


class EquiformerV2ModelWrapper(ModelWrapperBase[TConfig]):
    BackboneConfigType: TypeAlias = EquiformerV2Config
    BackboneType: TypeAlias = EquiformerV2Backbone
    BackboneOutputType: TypeAlias = EquiformerV2BackboneOutput

    class GraphBinaryClassificationHead(Base[TConfig], nn.Module):
        """Binary classification head for high uptake"""
        @override
        def __init__(
            self,
            config: TConfig,
            backbone: EquiformerV2Backbone,
            *,
            reduction: str = "mean",
        ):
            super().__init__(config)

            if reduction is None:
                reduction = self.config.graph_scalar_reduction_default

            # Output single logit for binary classification
            self.out_mlp = EqV2ScalarHead(
                backbone,
                hidden_channels_override=256,
                reduce=reduction,
                output_channels=1,
            )

        @override
        def forward(self, input: OutputHeadInput) -> torch.Tensor:
            data = input["data"]
            backbone_output = input["backbone_output"]

            output = self.out_mlp(data, backbone_output)
            logits = output["energy"]  # Shape: (B,)
            return logits

    class GraphScalarOutputHead(Base[TConfig], nn.Module): # type: ignore
        @override
        def __init__(
            self,
            config: TConfig,
            backbone: EquiformerV2Backbone,
            *,
            reduction: str = "sum",
            heteroscedastic: bool = False,
        ):
            super().__init__(config)

            self.heteroscedastic = heteroscedastic
            output_channels = 2 if heteroscedastic else 1  # mean + log_var OR just mean

            if reduction is None:
                reduction = self.config.graph_scalar_reduction_default

            self.out_mlp = EqV2ScalarHead(backbone,
                                          hidden_channels_override=256,
                                          reduce=reduction,
                                          output_channels=output_channels)
            # ----------------------------------
            # LATE FUSION
            # ----------------------------------
            self.mof_rbf_encoder= None
            self.late_fusion_mlp = None

            fusion_cfg = getattr(config, "mof_feature_fusion", None)
            if fusion_cfg and fusion_cfg.get("enabled", False) and fusion_cfg.get("fusion_type") == "late":
                in_dim = 1                                    # PLD or LCD is a scalar
                num_rbf = int(fusion_cfg.get("hidden_dim"))    # e.g., 128

                # scalar → RBF features (self-normalizing)
                self.mof_rbf_encoder = ScalarRBFEncoder(num_rbf=num_rbf)

                # RBF → scalar correction that we add to energy
                self.late_fusion_mlp = nn.Sequential(
                    nn.Linear(num_rbf, num_rbf),
                    nn.SiLU(),
                    nn.Linear(num_rbf, 1),
                )

        @override
        def forward(
            self,
            input: OutputHeadInput,
            *,
            scale: torch.Tensor | None = None,
            shift: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            data = input["data"]
            backbone_output = input["backbone_output"]

            output = self.out_mlp(data, backbone_output)
            energy = output["energy"]  # Shape: (B,) or (B, 2) if heteroscedastic

            # HETEROSCEDASTIC MODE
            if self.heteroscedastic:
                mean = energy[:, 0]
                log_var = energy[:, 1]

                if self.late_fusion_mlp is not None and self.mof_rbf_encoder is not None:
                    if not hasattr(data, "mof_descriptor"):
                        raise RuntimeError("Late fusion enabled but data.mof_descriptor missing")

                    global_feature = data.mof_descriptor
                    if global_feature.dim() == 1:
                        global_feature = global_feature.unsqueeze(-1)

                    global_feature = global_feature.to(mean.device)
                    rbf = self.mof_rbf_encoder(global_feature)
                    fused = self.late_fusion_mlp(rbf).squeeze(-1)
                    mean = mean + fused

                return mean, log_var  # Return tuple

            # STANDARD MODE
            if self.late_fusion_mlp is not None and self.mof_rbf_encoder is not None:
                if not hasattr(data, "mof_descriptor"):
                    raise RuntimeError("Late fusion enabled but data.mof_descriptor missing")

                global_feature = data.mof_descriptor
                # ensure shape is [B,1]
                if global_feature.dim() == 1:
                    global_feature = global_feature.unsqueeze(-1)

                global_feature = global_feature.to(energy.device)

                # Scalar → RBF features
                rbf = self.mof_rbf_encoder(global_feature)       # [B, num_rbf]
                # RBF → scalar bias, then add to energy
                fused = self.late_fusion_mlp(rbf).squeeze(-1)  # [B]
                energy = energy + fused

            return energy

    def __init__(self, config: TConfig):
        super().__init__(config)
        

    class GraphMulticlassClassificationOutputHead(
        Base[TConfig], nn.Module, Generic[TConfig]
    ):
        @override
        def __init__(
            self,
            config: TConfig,
            classification_config: MulticlassClassificationTargetConfig,
            backbone: EquiformerV2Backbone,
            reduction: str | None = None,
        ):
            super().__init__(config)

            if reduction is None:
                reduction = self.config.graph_scalar_reduction_default
            
            self.classification_config = classification_config
            self.reduction = reduction

            self.out_mlp = EqV2ClassificationHead(
                backbone, hidden_channels_override=256, reduce=reduction, 
                output_name="logits", 
                num_classes=classification_config.num_classes
                )
            

            self.dropout = None
            if classification_config.dropout:
                self.dropout = nn.Dropout(classification_config.dropout)

        @override
        def forward(self, input: OutputHeadInput) -> torch.Tensor:
            data = input["data"]
            x = input["backbone_output"]

            # TODO: Support dropout classification head for EqV2
            # if self.dropout is not None:
            #     x = self.dropout(x)

            output = self.out_mlp(data, x)  # (n, num_classes)
           
            return output["logits"]
        
    
    def load_backbone_state_dict(
        self,
        *,
        backbone: Mapping[str, Any],
        embedding: Mapping[str, Any],
        strict: bool = True,
    ):
        raise NotImplementedError("EquiformerV2PModelWrapper does not implement backbone loading")


class AutoregLlamaEquiformerV2ModelWrapper(ModelWrapperBase[TConfig]):
    BackboneConfigType: TypeAlias = AutoregressiveLlamaEquiformerV2Config
    BackboneType: TypeAlias = AutoregressiveLlamaEquiformerV2Backbone
    BackboneOutputType: TypeAlias = EquiformerV2BackboneOutput
   
    class GraphScalarOutputHead(Base[TConfig], nn.Module): # type: ignore
        @override
        def __init__(
            self,
            config: TConfig,
            backbone: EquiformerV2Backbone,
            *,
            reduction: str = "sum",
        ):
            super().__init__(config)

        
            if reduction is None:
                reduction = self.config.graph_scalar_reduction_default

            self.out_mlp = EqV2ScalarHead(backbone, 
                                          hidden_channels_override=256, 
                                          reduce=reduction)

        @override
        def forward(
            self,
            input: OutputHeadInput,
            *,
            scale: torch.Tensor | None = None,
            shift: torch.Tensor | None = None,
        ) -> torch.Tensor:
            data = input["data"]
            backbone_output = input["backbone_output"]

            output = self.out_mlp(data, backbone_output) 
            # output = rearrange(output["energy"], "b 1 -> b")
            return output["energy"]
        
    class GraphMulticlassClassificationOutputHead(
        Base[TConfig], nn.Module, Generic[TConfig]
    ):
        @override
        def __init__(
            self,
            config: TConfig,
            classification_config: MulticlassClassificationTargetConfig,
            backbone: AutoregressiveLlamaEquiformerV2Backbone,
            reduction: str | None = None,
        ):
            super().__init__(config)

            if reduction is None:
                reduction = self.config.graph_scalar_reduction_default
            
            self.classification_config = classification_config
            self.reduction = reduction

            self.out_mlp = EqV2ClassificationHead(
                backbone, hidden_channels_override=256, reduce=reduction, 
                output_name="logits", 
                num_classes=classification_config.num_classes
                )
            

            self.dropout = None
            if classification_config.dropout:
                self.dropout = nn.Dropout(classification_config.dropout)

        @override
        def forward(self, input: OutputHeadInput) -> torch.Tensor:
            data = input["data"]
            x = input["backbone_output"]

            # TODO: Support dropout classification head for EqV2
            # if self.dropout is not None:
            #     x = self.dropout(x)

            output = self.out_mlp(data, x)  # (n, num_classes)
           
            return output["logits"]

    def __init__(self, config: TConfig):
        super().__init__(config)

    def load_backbone_state_dict(
        self,
        *,
        backbone: Mapping[str, Any],
        embedding: Mapping[str, Any],
        strict: bool = True,
    ):
        raise NotImplementedError("EquiformerV2PModelWrapper does not implement backbone loading")

def __test(_int: int):
    pass