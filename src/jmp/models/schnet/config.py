"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
from jmp.lightning import TypedConfig
from typing_extensions import override


class SchNetBackboneConfig(TypedConfig):
    name: Literal["schnet", "SchNet"] = "schnet"
    num_targets: int = 1
    use_pbc: bool = False
    use_pbc_single: bool = False
    regress_forces: bool = True
    direct_forces: bool = False
    otf_graph: bool = False
    num_elements: int = 100
    embedding_dim: int = 128
    num_filters: int = 128
    num_interactions: int = 6
    num_gaussians: int = 50
    cutoff: float = 10.0
    readout: str = "add"
    
    "The following are not used for this backbone but required pydantic __post_init__ validation"
    qint_tags: list[int] = [1, 2]
    activation: str = "scaled_silu"
    dropout: float | None = None
    edge_dropout: float | None = None  
    num_blocks: int = 0
    
    @property
    def hidden_channels(self):
        return self.embedding_dim
    
    @property
    def num_embeddings(self):
        return self.num_elements

    @property
    def emb_size_atom(self):
        return self.embedding_dim

    @classmethod
    def base(cls):
        return cls()

    # @classmethod
    # def download_base(cls, **kwargs):
    #     """Load SchNet config from github"""
    #     import requests
    #     import yaml

    #     # read config from url
    #     response = requests.get(
    #         "https://raw.githubusercontent.com/Open-Catalyst-Project/ocp/main/configs/s2ef/all/schnet/schnet.yml"
    #     )
    #     config_dict = yaml.safe_load(response.text)

    #     model_config: dict = {**config_dict["model"]}
    #     _ = model_config.pop("name", None)
    #     _ = model_config.pop("scale_file", None)

    #     for key in list(model_config.keys()):
    #         if any([key.startswith(prefix) for prefix in ["cutoff", "max_neighbors"]]):
    #             _ = model_config.pop(key)

    #     model_config.update(kwargs)
    #     config = cls.from_dict(model_config)
    #     return config


    @classmethod
    def from_ckpt(
        cls,
        ckpt_path: Path | str,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = ckpt["config"]
        if transform is not None:
            config = transform(config)
        return cls(**config)



