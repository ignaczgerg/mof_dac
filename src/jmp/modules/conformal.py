"""
Conformal Prediction Module for Uncertainty Quantification.

This module provides conformal prediction methods for both regression and
classification tasks, with a focus on rare event detection (e.g., high CO2 uptake).

Conformal prediction provides distribution-free, finite-sample valid prediction
sets/intervals with guaranteed coverage, making it particularly suitable for
OOD (out-of-distribution) scenarios.

References:
- Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World.
- Romano, Patterson, Candès (2019). Conformalized Quantile Regression.
- Angelopoulos & Bates (2021). A Gentle Introduction to Conformal Prediction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Any
from torch_geometric.data.data import BaseData
from torch_geometric.data.batch import Batch


@dataclass
class ConformalResult:
    """Result from conformal prediction."""

    # For regression: prediction intervals
    lower_bound: Optional[torch.Tensor] = None
    upper_bound: Optional[torch.Tensor] = None
    point_prediction: Optional[torch.Tensor] = None

    # For classification: prediction sets
    prediction_set: Optional[torch.Tensor] = None  # Binary mask of included classes
    predicted_class: Optional[torch.Tensor] = None
    class_probabilities: Optional[torch.Tensor] = None

    # Uncertainty metrics
    interval_width: Optional[torch.Tensor] = None
    set_size: Optional[torch.Tensor] = None

    # Conformal scores (for analysis)
    nonconformity_scores: Optional[torch.Tensor] = None


@dataclass
class ConformalConfig:
    """Configuration for conformal prediction."""

    # Coverage level (1 - alpha)
    alpha: float = 0.1  # 90% coverage by default

    # Method for computing nonconformity scores
    score_type: Literal["absolute", "normalized", "quantile"] = "absolute"

    # For classification
    classification_method: Literal["lac", "aps", "raps"] = "aps"
    # lac: Least Ambiguous set-valued Classifier
    # aps: Adaptive Prediction Sets
    # raps: Regularized Adaptive Prediction Sets

    # RAPS regularization parameters
    raps_k_reg: int = 5  # Number of classes to start regularizing
    raps_lambda: float = 0.01  # Regularization strength

    # Whether to use Bonferroni correction for multiple comparisons
    bonferroni_correction: bool = False


class ConformalRegressor:
    """
    Conformal prediction for regression tasks.

    Provides prediction intervals with guaranteed coverage for any
    underlying regression model. Particularly useful for:
    - Rare event detection (e.g., high CO2 uptake structures)
    - OOD detection (wide intervals indicate uncertainty)
    - Risk-aware decision making

    Usage:
        # 1. Train your regression model
        model = train_model(train_data)

        # 2. Calibrate conformal predictor on held-out data
        conformal = ConformalRegressor(config=ConformalConfig(alpha=0.1))
        conformal.calibrate(model, calibration_loader)

        # 3. Make predictions with intervals
        result = conformal.predict(model, test_data)
        # result.lower_bound, result.upper_bound have 90% coverage guarantee
    """

    def __init__(self, config: Optional[ConformalConfig] = None):
        self.config = config or ConformalConfig()
        self.calibrated = False
        self.quantile: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None

    def calibrate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        predicted_std: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Calibrate the conformal predictor using held-out calibration data.

        Args:
            predictions: Model predictions on calibration set (N,)
            targets: Ground truth values (N,)
            predicted_std: Optional predicted standard deviations for normalized scores (N,)
        """
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        # Compute nonconformity scores
        if self.config.score_type == "absolute":
            scores = np.abs(predictions - targets)
        elif self.config.score_type == "normalized":
            if predicted_std is None:
                raise ValueError("predicted_std required for normalized scores")
            std = predicted_std.detach().cpu().numpy()
            scores = np.abs(predictions - targets) / (std + 1e-8)
        else:
            raise ValueError(f"Unknown score_type: {self.config.score_type}")

        self.calibration_scores = scores

        # Compute quantile for desired coverage
        n = len(scores)
        # Finite-sample correction: (n+1)(1-alpha)/n quantile
        q_level = np.ceil((n + 1) * (1 - self.config.alpha)) / n
        q_level = min(q_level, 1.0)

        self.quantile = np.quantile(scores, q_level)
        self.calibrated = True

        # Store calibration statistics
        self.calibration_stats = {
            "n_calibration": n,
            "quantile_level": q_level,
            "quantile_value": self.quantile,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "max_score": float(np.max(scores)),
        }

    def predict(
        self,
        predictions: torch.Tensor,
        predicted_std: Optional[torch.Tensor] = None,
    ) -> ConformalResult:
        """
        Generate prediction intervals with guaranteed coverage.

        Args:
            predictions: Model predictions (N,)
            predicted_std: Optional predicted standard deviations (N,)

        Returns:
            ConformalResult with lower_bound, upper_bound, and interval_width
        """
        if not self.calibrated:
            raise RuntimeError("Conformal predictor must be calibrated first")

        device = predictions.device

        if self.config.score_type == "normalized" and predicted_std is not None:
            # Normalized conformal intervals
            margin = self.quantile * predicted_std
        else:
            # Absolute conformal intervals
            margin = torch.full_like(predictions, self.quantile)

        lower = predictions - margin
        upper = predictions + margin

        return ConformalResult(
            lower_bound=lower,
            upper_bound=upper,
            point_prediction=predictions,
            interval_width=upper - lower,
        )

    def check_coverage(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        predicted_std: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """
        Check empirical coverage on test data.

        Returns:
            Dictionary with coverage statistics
        """
        result = self.predict(predictions, predicted_std)

        in_interval = (targets >= result.lower_bound) & (targets <= result.upper_bound)
        coverage = in_interval.float().mean().item()

        return {
            "empirical_coverage": coverage,
            "target_coverage": 1 - self.config.alpha,
            "mean_interval_width": result.interval_width.mean().item(),
            "median_interval_width": result.interval_width.median().item(),
        }


class ConformalClassifier:
    """
    Conformal prediction for binary classification tasks.

    Provides prediction sets with guaranteed coverage. For rare event
    detection (like high CO2 uptake), this is particularly valuable:
    - A structure is flagged if the "high uptake" class is in the prediction set
    - Guaranteed false negative rate control

    Usage:
        # 1. Train your classification model
        model = train_model(train_data)

        # 2. Calibrate on held-out data
        conformal = ConformalClassifier(config=ConformalConfig(alpha=0.05))
        conformal.calibrate(model_logits, true_labels)

        # 3. Make predictions
        result = conformal.predict(test_logits)
        # result.prediction_set indicates which classes are in the set
        # For binary: if prediction_set[:, 1] == 1, "high uptake" is possible
    """

    def __init__(self, config: Optional[ConformalConfig] = None):
        self.config = config or ConformalConfig()
        self.calibrated = False
        self.threshold: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None

    def _compute_aps_scores(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Adaptive Prediction Set (APS) scores.

        APS score = sum of probabilities until true class is included + U * p(y)
        where U ~ Uniform(0, 1) for randomization.

        For binary classification, this simplifies significantly.
        """
        n = len(labels)

        if probabilities.ndim == 1:
            # Binary case: probabilities is P(Y=1)
            prob_pos = probabilities
            prob_neg = 1 - prob_pos

            scores = np.zeros(n)
            for i in range(n):
                if labels[i] == 1:
                    # True class is positive
                    if prob_pos[i] >= prob_neg[i]:
                        # Positive class is most likely
                        u = np.random.uniform(0, 1)
                        scores[i] = u * prob_pos[i]
                    else:
                        # Negative class is more likely
                        scores[i] = prob_neg[i] + np.random.uniform(0, 1) * prob_pos[i]
                else:
                    # True class is negative
                    if prob_neg[i] >= prob_pos[i]:
                        u = np.random.uniform(0, 1)
                        scores[i] = u * prob_neg[i]
                    else:
                        scores[i] = prob_pos[i] + np.random.uniform(0, 1) * prob_neg[i]
        else:
            # Multi-class case
            n_classes = probabilities.shape[1]
            scores = np.zeros(n)

            for i in range(n):
                probs = probabilities[i]
                true_label = int(labels[i])

                # Sort classes by probability (descending)
                sorted_idx = np.argsort(-probs)

                # Cumulative probability until true class
                cumsum = 0
                for j, idx in enumerate(sorted_idx):
                    if idx == true_label:
                        u = np.random.uniform(0, 1)
                        scores[i] = cumsum + u * probs[idx]
                        break
                    cumsum += probs[idx]

        return scores

    def _compute_lac_scores(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Least Ambiguous set-valued Classifier (LAC) scores.

        LAC score = 1 - P(true class)
        """
        if probabilities.ndim == 1:
            # Binary case
            prob_true = np.where(labels == 1, probabilities, 1 - probabilities)
        else:
            # Multi-class case
            prob_true = probabilities[np.arange(len(labels)), labels.astype(int)]

        return 1 - prob_true

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Calibrate the conformal classifier.

        Args:
            logits: Model logits (N,) for binary or (N, C) for multi-class
            labels: True labels (N,)
        """
        # Convert to probabilities
        if logits.dim() == 1:
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        else:
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        labels_np = labels.detach().cpu().numpy()

        # Compute nonconformity scores
        if self.config.classification_method == "aps":
            scores = self._compute_aps_scores(probabilities, labels_np)
        elif self.config.classification_method == "lac":
            scores = self._compute_lac_scores(probabilities, labels_np)
        else:
            raise ValueError(f"Unknown method: {self.config.classification_method}")

        self.calibration_scores = scores

        # Compute threshold for desired coverage
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.config.alpha)) / n
        q_level = min(q_level, 1.0)

        self.threshold = np.quantile(scores, q_level)
        self.calibrated = True

        self.calibration_stats = {
            "n_calibration": n,
            "quantile_level": q_level,
            "threshold": self.threshold,
            "mean_score": float(np.mean(scores)),
            "positive_rate": float(np.mean(labels_np)),
        }

    def predict(
        self,
        logits: torch.Tensor,
    ) -> ConformalResult:
        """
        Generate prediction sets with guaranteed coverage.

        Args:
            logits: Model logits (N,) for binary or (N, C) for multi-class

        Returns:
            ConformalResult with prediction_set indicating included classes
        """
        if not self.calibrated:
            raise RuntimeError("Conformal classifier must be calibrated first")

        device = logits.device

        # Convert to probabilities
        if logits.dim() == 1:
            prob_pos = torch.sigmoid(logits)
            probabilities = torch.stack([1 - prob_pos, prob_pos], dim=-1)
        else:
            probabilities = torch.softmax(logits, dim=-1)

        n_samples, n_classes = probabilities.shape

        # For each sample, include classes until cumulative prob >= threshold
        # This creates prediction sets with guaranteed coverage

        # Sort classes by probability (descending)
        sorted_probs, sorted_idx = torch.sort(probabilities, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Include classes until we exceed threshold
        # Add small randomization for exact coverage
        u = torch.rand(n_samples, 1, device=device)
        include_mask = cumsum - sorted_probs * u <= self.threshold

        # Convert back to original class ordering
        prediction_set = torch.zeros_like(probabilities, dtype=torch.bool)
        for i in range(n_samples):
            for j in range(n_classes):
                if include_mask[i, j]:
                    prediction_set[i, sorted_idx[i, j]] = True

        # For binary, also compute predicted class
        if n_classes == 2:
            predicted_class = (probabilities[:, 1] > 0.5).long()
        else:
            predicted_class = probabilities.argmax(dim=-1)

        return ConformalResult(
            prediction_set=prediction_set,
            predicted_class=predicted_class,
            class_probabilities=probabilities,
            set_size=prediction_set.float().sum(dim=-1),
        )

    def predict_high_uptake(
        self,
        logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Specialized prediction for high uptake detection.

        Returns:
            Dictionary with:
            - "possibly_high": True if high uptake class is in prediction set
            - "definitely_high": True if only high uptake class is in set
            - "probability": Predicted probability of high uptake
            - "set_size": Size of prediction set (1 = certain, 2 = uncertain)
        """
        result = self.predict(logits)

        # For binary classification, class 1 is "high uptake"
        possibly_high = result.prediction_set[:, 1]  # High uptake in prediction set
        definitely_high = possibly_high & ~result.prediction_set[:, 0]  # Only high uptake

        return {
            "possibly_high": possibly_high,
            "definitely_high": definitely_high,
            "probability": result.class_probabilities[:, 1],
            "set_size": result.set_size,
            "uncertain": result.set_size > 1,
        }

    def check_coverage(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        """
        Check empirical coverage on test data.
        """
        result = self.predict(logits)

        # Coverage: true label is in prediction set
        if logits.dim() == 1:
            covered = torch.where(
                labels == 1,
                result.prediction_set[:, 1],
                result.prediction_set[:, 0],
            )
        else:
            covered = result.prediction_set.gather(1, labels.long().unsqueeze(1)).squeeze(1)

        coverage = covered.float().mean().item()

        # Stratified coverage for rare class
        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_coverage = covered[pos_mask].float().mean().item() if pos_mask.any() else float('nan')
        neg_coverage = covered[neg_mask].float().mean().item() if neg_mask.any() else float('nan')

        return {
            "empirical_coverage": coverage,
            "target_coverage": 1 - self.config.alpha,
            "positive_class_coverage": pos_coverage,
            "negative_class_coverage": neg_coverage,
            "mean_set_size": result.set_size.mean().item(),
            "empty_set_rate": (result.set_size == 0).float().mean().item(),
        }


def calibrate_from_dataloader(
    model: nn.Module,
    dataloader: Any,
    config: ConformalConfig,
    target_key: str = "co2_uptake",
    classification: bool = False,
    device: str = "cuda",
) -> ConformalRegressor | ConformalClassifier:
    """
    Convenience function to calibrate conformal predictor from a dataloader.

    Args:
        model: Trained model
        dataloader: Calibration data loader
        config: Conformal configuration
        target_key: Name of target property
        classification: If True, calibrate classifier; else regressor
        device: Device for computation

    Returns:
        Calibrated conformal predictor
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_stds = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            preds = model(batch)

            if classification:
                cls_key = f"{target_key}_cls"
                if cls_key in preds:
                    all_preds.append(preds[cls_key])
            else:
                all_preds.append(preds[target_key])

                # Check for heteroscedastic predictions
                log_var_key = f"{target_key}_log_var"
                if log_var_key in preds:
                    std = torch.exp(0.5 * preds[log_var_key])
                    all_stds.append(std)

            all_targets.append(getattr(batch, target_key))

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if classification:
        conformal = ConformalClassifier(config)
        # For classification, targets should be binary
        conformal.calibrate(all_preds, all_targets)
    else:
        conformal = ConformalRegressor(config)
        all_stds = torch.cat(all_stds, dim=0) if all_stds else None
        conformal.calibrate(all_preds, all_targets, all_stds)

    return conformal
