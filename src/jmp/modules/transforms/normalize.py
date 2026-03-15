"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections.abc import Mapping
from typing import cast, Literal
from pydantic import Field


import numpy as np
import torch
from jmp.lightning import TypedConfig
from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar

T = TypeVar("T", float, torch.Tensor, np.ndarray, infer_variance=True)


def _process_value(value: T) -> torch.Tensor:
    return cast(
        torch.Tensor,
        torch.tensor(value) if not torch.is_tensor(value) else value,
    )

class PositionNormalizationConfig(TypedConfig):
    """
    Handles both standard and log-based normalization.
    
    Attributes:
        mean (float): Mean used for standardization.
        std (float): Standard deviation used for standardization.
        normalization_type (str): Either 'standard' or 'log'.
    """
    
    type: Literal["pos"] = Field(default="pos")
    mean: float = 1.0
    std: float = 1.0
    normalization_type: Literal["standard", "log"] = "standard"

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        if self.normalization_type == "log":
            value = torch.log1p(value)  # Apply log transformation
        
        center = value.mean(dim=0, keepdim=True)
        value -= center
        return (value - self.mean) / self.std

    def denormalize(self, value: T, center: T = None) -> T:
        value = (value * self.std) + self.mean
        if center is not None:
            value += center

        if self.normalization_type == "log":
            value = torch.expm1(value)  # Reverse log transformation
        
        return value

class NormalizationConfig(TypedConfig):
    """
    Handles both standard and log-based normalization.
    
    Attributes:
        mean (float): Mean used for standardization.
        std (float): Standard deviation used for standardization.
        normalization_type (str): Either 'standard' or 'log'.
    """
    type: Literal["default"] = Field(default="default")
    mean: float = 1.0
    std: float = 1.0
    normalization_type: Literal["standard", "log"] = "standard"

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        import sys
        # print(f"DEBUG NORMALIZE METHOD: type={self.normalization_type}, mean={self.mean:.6f}, std={self.std:.6f}", file=sys.stderr, flush=True)
        
        original_value = value.clone()
        if self.normalization_type == "log":
            eps = 1e-12
            # print(f"DEBUG: BEFORE LOG - value: {value.item():.6f}", file=sys.stderr, flush=True)
            value = torch.abs(value)
            value = torch.clamp(value, min=eps)
            # print(f"DEBUG: AFTER ABS - value: {value.item():.6f}", file=sys.stderr, flush=True)
            value = torch.log(value)  
            # print(f"DEBUG: AFTER LOG - value: {value.item():.6f}", file=sys.stderr, flush=True)
        
        result = (value - self.mean) / self.std
        # print(f"DEBUG: FINAL RESULT: {result.item():.6f}", file=sys.stderr, flush=True)
        return result

    def denormalize(self, value: T) -> T:
        value = (value * self.std) + self.mean
        if self.normalization_type == "log":
            max_log = torch.log(torch.tensor(torch.finfo(
                value.dtype if torch.is_tensor(value) else torch.float32
            ).max, device=value.device if torch.is_tensor(value) else None)) - 1.0
            value = torch.clamp(value, max=max_log)
            value = torch.exp(value)
        return value


def normalize(properties: Mapping[str, NormalizationConfig]):
    def _normalize(data: BaseData):
        nonlocal properties

        # print(f"DEBUG: _normalize called with properties: {list(properties.keys())}")
        for key, d in properties.items():
            if (value := getattr(data, key, None)) is None:
                raise ValueError(f"Property {key} not found in data")

            # print("===============  DEBUG: normalize  ==================")
            # print(f"DEBUG: Normalizing {key} - Original value: {value.item():.6f}")
            # print("===============  DEBUG: normalize  ==================")
            value = _process_value(value)
            value = d.normalize(value)
            # print("===============  DEBUG: normalize  ==================")
            # print(f"DEBUG: Normalizing {key} - After normalization: {value.item():.6f}")
            # print("===============  DEBUG: normalize  ==================")
            setattr(data, key, value)
            setattr(data, f"{key}_norm_mean", torch.full_like(value, d.mean))
            setattr(data, f"{key}_norm_std", torch.full_like(value, d.std))

        return data

    return _normalize


def denormalize_tensor(value: torch.Tensor, norm: NormalizationConfig) -> torch.Tensor:
    """
    Applies denormalization to a single tensor based on the provided NormalizationConfig.
    
    Args:
        value (torch.Tensor): The normalized tensor to be denormalized.
        config (NormalizationConfig): The normalization configuration used during training.
    
    Returns:
        torch.Tensor: The denormalized tensor.
    """
    value = _process_value(value)
    return norm.denormalize(value)


def denormalize_batch(
    batch: BaseData,
    additional_tensors: dict[str, torch.Tensor] | None = None,
):
    if additional_tensors is None:
        additional_tensors = {}

    keys: set[str] = set(batch.keys())

    # find all keys that have a denorm_mean and denorm_std
    norm_keys: set[str] = {
        key.replace("_norm_mean", "") for key in keys if key.endswith("_norm_mean")
    } & {key.replace("_norm_std", "") for key in keys if key.endswith("_norm_std")}

    for key in norm_keys:
        mean = getattr(batch, f"{key}_norm_mean")
        std = getattr(batch, f"{key}_norm_std")
        value = getattr(batch, key)

        # TODO: Support log normalization
        value = (value * std) + mean 
        
        if key == 'pos':
            # If pos is denormalized, we also need to adjust the center_pos
            center_pos = getattr(batch, 'center', None)
            if center_pos is not None:
                value = value + center_pos[batch.batch]
        
        setattr(batch, key, value)

        if (additional_value := additional_tensors.pop(key, None)) is not None:
            additional_tensors[key] = (additional_value * std) + mean
            if key == 'pos':
                # If pos is denormalized, we also need to adjust the center_pos
                center_pos = getattr(batch, 'center', None)
                if center_pos is not None:
                    additional_tensors[key] = additional_tensors[key] + center_pos[batch.batch]

    return batch, additional_tensors
