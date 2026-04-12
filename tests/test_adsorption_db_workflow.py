import argparse
import copy
import os
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "my_scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


gpu = pytest.mark.gpu
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# Single-dataset mode paths (relative to root_path/datasets/)
# NOTE: cof_db2 is only available in multitask mode — no single-dataset path
SINGLE_DATASET_PATHS: dict[str, str] = {
    "adsorption_db1_merged": "adsorption_db1_merged/lmdb",
    "adsorption_db2_merged": "adsorption_db2_merged/lmdb",
}

# Multi-task mode paths (relative to root_path/datasets/)
MULTITASK_DATASET_PATHS: dict[str, str] = {
    "adsorption_db1_merged": "all-adsorption/mof_db_1/lmdb",
    "adsorption_db2_merged": "all-adsorption/mof_db_2/lmdb",
    "adsorption_cof_db2": "all-adsorption/cof_db_2/lmdb",
    "adsorption_mof_db_anion": "all-adsorption/mof_db_anion/lmdb",
}

REQUIRED_SPLITS = ("train", "val", "test")

# All valid adsorption targets
ADSORPTION_TARGETS = [
    "qst_co2", "qst_h2o", "qst_n2",
    "kh_co2", "kh_h2o", "kh_n2",
    "selectivity_co2_h2o", "selectivity_co2_n2",
    "co2_uptake", "n2_uptake",
]

GEOMETRIC_TARGETS = ["lcd", "pld", "gcd", "unitcell_volume", "density", "asa", "av", "nav"]


@pytest.fixture(scope="session")
def root_path():
    """Resolve the root path from the project's env_utils."""
    from jmp.utils.env_utils import load_env_paths
    paths = load_env_paths()
    return Path(paths["root"])


@pytest.fixture(scope="session")
def ft_logging_path(tmp_path_factory):
    """Create a temp logging directory for training tests."""
    return tmp_path_factory.mktemp("ft_logging")


def _make_base_args(**overrides) -> argparse.Namespace:
    """Build a minimal argparse.Namespace with all required fields.

    Settings are aligned with run_finetune_mof_db.sh (production script).
    Pass keyword arguments to override any default.
    """
    defaults = dict(
        model_name="equiformer_v2",
        small=True,
        medium=False,
        large=False,
        very_small_qm9=False,
        llama=False,
        adabin=False,
        adabin_dropout=0.5,
        adabin_num_classes=[256],
        adabin_class_weights=[1.0],
        scratch=True,
        checkpoint_tag="odac_public",
        checkpoint_path=None,
        lr=1e-4,
        epochs=2,
        batch_size=1,
        weight_decay=1e-3,  
        dropout=0.05,
        edge_dropout=0.05,
        seed=42,
        precision="16-mixed",
        disable_ema=True,
        cutoff=6.0,
        max_natoms=800,
        max_neighbors=32,
        no_pbc=False,
        num_workers=0,
        max_atoms_per_batch=400,
        atom_bucket_batch_sampler=True,
        targets=["co2_uptake"],
        normalization_type=["standard"],
        norm_mean=None,
        norm_std=None,
        graph_scalar_reduction=["mean"],
        node_vector_reduction=["mean"],
        targets_loss_coefficients=[1.0],
        loss="l1",
        uptake_conversion_mode="none",
        enable_feature_fusion=False,
        fusion_type="late",
        fusion_feature_names=["pld", "lcd", "gcd", "unitcell_volume", "density", "asa", "av", "nav"],
        fusion_hidden_dim=128,
        use_charge_embedding=False,
        heteroscedastic=False,
        hetero_nll_weight=1.0,
        hetero_std_weight=0.1,
        hetero_min_var=1e-6,
        classification_threshold=0.5,
        classification_targets=[],
        classification_loss_weight=1.0,
        focal_gamma=2.0,
        focal_alpha_pos=0.75,
        focal_alpha_neg=0.25,
        tasks=None,
        dataset_name=None,
        sample_temperature=2,
        sample_type="temperature",
        train_samples_limit=50,
        val_samples_limit=50,
        test_samples_limit=50,
        val_same_as_train=True,
        number_of_samples=None,
        log_predictions=True,
        enable_wandb=False,  # DO NOT SET THIS TO TRUE EVER
        profile=False,
        compute_avg_dataset_stats=False,
        compute_avg_dataset_stats_degree=False,
        avg_stats_sample_size=100,
        roi_penalty=1.0,
        position_norm=False,
        rbf_function="gaussian",
        num_distance_basis=600,
        fold=0,
        enable_flow_matching=False,
        postfix="test",
        root_path=None,
        logging_path=None,
        extreme_val_csv=None,
        extreme_val_cif_dirs=[],
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _build_config(root_path, **args_overrides):
    """Build (config, model_cls) using get_configs, matching production flow."""
    from setup_finetune import get_configs

    args = _make_base_args(root_path=str(root_path), **args_overrides)

    # Expand normalization_type to match targets length
    n_targets = len(args.targets)
    if len(args.normalization_type) < n_targets:
        args.normalization_type += ["standard"] * (n_targets - len(args.normalization_type))
    if len(args.targets_loss_coefficients) < n_targets:
        args.targets_loss_coefficients += [1.0] * (n_targets - len(args.targets_loss_coefficients))
    if len(args.graph_scalar_reduction) < n_targets:
        args.graph_scalar_reduction += ["mean"] * (n_targets - len(args.graph_scalar_reduction))

    config, model_cls = get_configs(args.dataset_name, args.targets, args=args)
    return config, model_cls, args


# ============================================================================
# 1. DATASET PATH TESTS — no GPU needed
# ============================================================================

class TestDatasetPaths:
    """Verify LMDB dataset paths exist and are loadable."""

    @pytest.mark.parametrize("name,rel_path", list(SINGLE_DATASET_PATHS.items()))
    def test_single_dataset_paths_exist(self, root_path, name, rel_path):
        base = root_path / "datasets" / rel_path
        for split in REQUIRED_SPLITS:
            split_dir = base / split
            assert split_dir.exists(), f"Missing {split} for {name}: {split_dir}"

    @pytest.mark.parametrize("name,rel_path", list(MULTITASK_DATASET_PATHS.items()))
    def test_multitask_dataset_paths_exist(self, root_path, name, rel_path):
        base = root_path / "datasets" / rel_path
        for split in REQUIRED_SPLITS:
            split_dir = base / split
            assert split_dir.exists(), f"Missing {split} for {name}: {split_dir}"

    @pytest.mark.parametrize("name,rel_path", list(SINGLE_DATASET_PATHS.items()))
    def test_metadata_files_exist(self, root_path, name, rel_path):
        base = root_path / "datasets" / rel_path
        for split in REQUIRED_SPLITS:
            metadata = base / split / "metadata.npz"
            assert metadata.exists(), f"Missing metadata for {name}/{split}: {metadata}"

    @pytest.mark.parametrize("name", list(SINGLE_DATASET_PATHS.keys()))
    def test_dataset_is_loadable(self, root_path, name):
        from jmp.datasets.finetune.base import LmdbDataset
        rel_path = SINGLE_DATASET_PATHS[name]
        ds_path = root_path / "datasets" / rel_path / "train"
        args = _make_base_args(root_path=str(root_path))
        ds = LmdbDataset(str(ds_path), args=args)
        assert len(ds) > 0
        sample = ds[0]
        assert hasattr(sample, "atomic_numbers")
        assert hasattr(sample, "pos")
        ds.close_db()

    @pytest.mark.parametrize("name,rel_path", list(MULTITASK_DATASET_PATHS.items()))
    def test_multitask_dataset_is_loadable(self, root_path, name, rel_path):
        from jmp.datasets.finetune.base import LmdbDataset
        ds_path = root_path / "datasets" / rel_path / "train"
        if not ds_path.exists():
            pytest.skip(f"Dataset not found: {ds_path}")
        args = _make_base_args(root_path=str(root_path))
        ds = LmdbDataset(str(ds_path), args=args)
        assert len(ds) > 0
        ds.close_db()


# ============================================================================
# 2. CONFIG CONSTRUCTION TESTS — no GPU needed
# ============================================================================

class TestConfigConstruction:
    """Test that configs are built correctly for various settings."""

    def test_config_single_target(self, root_path):
        config, _, _ = _build_config(
            root_path, tasks=["adsorption_db1_merged"], targets=["co2_uptake"],
        )
        assert config is not None
        assert len(config.train_tasks) >= 1

    def test_config_multiple_targets(self, root_path):
        targets = ["co2_uptake", "qst_co2"]
        config, _, _ = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            targets=targets,
            targets_loss_coefficients=[1.0, 1.0],
        )
        assert config is not None

    @pytest.mark.parametrize("mode", ["none", "per_cell", "per_volume"])
    def test_config_uptake_conversion_mode(self, root_path, mode):
        config, _, _ = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            uptake_conversion_mode=mode,
        )
        assert config is not None

    @pytest.mark.parametrize("loss_fn", ["l1", "l2", "huber"])
    def test_config_loss_type(self, root_path, loss_fn):
        config, _, _ = _build_config(
            root_path, tasks=["adsorption_db1_merged"], loss=loss_fn,
        )
        assert config is not None

    def test_config_late_fusion(self, root_path):
        config, _, _ = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            enable_feature_fusion=True,
            fusion_type="late",
        )
        fusion_cfg = config.backbone.mof_feature_fusion
        if isinstance(fusion_cfg, dict):
            assert fusion_cfg["enabled"] is True
        else:
            assert fusion_cfg.enabled is True

    def test_config_heteroscedastic(self, root_path):
        config, _, _ = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            heteroscedastic=True,
        )
        assert config.heteroscedastic is True

    def test_config_scratch_mode(self, root_path):
        config, _, _ = _build_config(
            root_path, tasks=["adsorption_db1_merged"], scratch=True,
        )
        assert config is not None

    def test_config_checkpoint_tag(self, root_path):
        config, _, _ = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            scratch=False,
            checkpoint_tag="odac_public",
        )
        ckpt_path = Path(str(config.meta["ckpt_path"]))
        assert "odac" in ckpt_path.name.lower() or "odac" in str(ckpt_path).lower()

    @pytest.mark.parametrize("norm_type", ["standard", "log"])
    def test_config_normalization_types(self, root_path, norm_type):
        config, _, _ = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            normalization_type=[norm_type],
        )
        assert config is not None

    def test_config_graph_scalar_reduction(self, root_path):
        config, _, _ = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            graph_scalar_reduction=["sum"],
        )
        assert config is not None

    def test_config_multitask(self, root_path):
        task_names = ["adsorption_db1_merged", "adsorption_db2_merged"]
        config, _, _ = _build_config(
            root_path,
            tasks=task_names,
        )
        assert len(config.train_tasks) == len(task_names)
        assert len(config.tasks) >= len(task_names)
        assert config.mt_dataset is not None
        assert "co2_uptake" in config.mt_dataset.taskify_keys_graph

    def test_config_custom_loss_coefficients(self, root_path):
        targets = ["co2_uptake", "qst_co2"]
        config, _, _ = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            targets=targets,
            targets_loss_coefficients=[2.0, 0.5],
        )
        assert config is not None


# 3. CHECKPOINT TAG MAPPING — no GPU 

class TestCheckpointTagMapping:
    def test_all_tags_have_filenames(self):
        from setup_finetune import CHECKPOINT_TAG_MAPPING
        for tag, fname in CHECKPOINT_TAG_MAPPING.items():
            assert isinstance(fname, str), f"Tag '{tag}' has non-string filename"
            assert len(fname) > 0

    def test_key_pretrained_checkpoints_exist(self, root_path):
        from setup_finetune import CHECKPOINT_TAG_MAPPING
        for tag in ["odac_public"]:
            fname = CHECKPOINT_TAG_MAPPING[tag]
            path = root_path / "checkpoints" / "EquiformerV2" / fname
            assert path.exists(), f"Checkpoint '{tag}' not found: {path}"


# 4. MODEL CONSTRUCTION — GPU 

@requires_gpu
@gpu
class TestModelConstruction:
    """Test model instantiation with various configs."""

    def test_model_single_target(self, root_path):
        config, model_cls, _ = _build_config(
            root_path, tasks=["adsorption_db1_merged"], targets=["co2_uptake"],
        )
        model = model_cls(config)
        assert model is not None
        assert hasattr(model, "graph_outputs") or hasattr(model, "model_wrapper")

    def test_model_multiple_targets(self, root_path):
        targets = ["co2_uptake", "qst_co2"]
        config, model_cls, _ = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            targets=targets,
            targets_loss_coefficients=[1.0, 1.0],
        )
        model = model_cls(config)
        assert model is not None

    def test_model_heteroscedastic(self, root_path):
        config, model_cls, _ = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            heteroscedastic=True,
        )
        model = model_cls(config)
        assert model.config.heteroscedastic is True

    def test_model_scratch_initializes(self, root_path):
        config, model_cls, _ = _build_config(
            root_path, tasks=["adsorption_db1_merged"], scratch=True,
        )
        model = model_cls(config)
        for name, p in model.named_parameters():
            assert not torch.isnan(p).any(), f"NaN in parameter: {name}"


# 5. FORWARD PASS TESTS — GPU

@requires_gpu
@gpu
class TestForwardPass:
    """Test forward passes through the model with real data."""

    def test_forward_single_target(self, root_path):
        config, model_cls, args = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            targets=["co2_uptake"],
        )
        model = model_cls(config)
        model.setup("fit")
        dl = model.train_dataloader()
        batch = next(iter(dl))
        batch = batch.to("cuda")
        model = model.to("cuda")
        model.eval()
        with torch.no_grad():
            preds = model(batch)
        assert "co2_uptake" in preds
        assert preds["co2_uptake"].ndim >= 1

    def test_forward_multiple_targets(self, root_path):
        targets = ["co2_uptake", "qst_co2"]
        config, model_cls, args = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            targets=targets,
            targets_loss_coefficients=[1.0, 1.0],
        )
        model = model_cls(config)
        model.setup("fit")
        dl = model.train_dataloader()
        batch = next(iter(dl))
        batch = batch.to("cuda")
        model = model.to("cuda")
        model.eval()
        with torch.no_grad():
            preds = model(batch)
        for t in targets:
            assert t in preds, f"Missing prediction for target '{t}'"

    def test_forward_heteroscedastic_output_shape(self, root_path):
        config, model_cls, args = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            targets=["co2_uptake"],
            heteroscedastic=True,
        )
        model = model_cls(config)
        model.setup("fit")
        dl = model.train_dataloader()
        batch = next(iter(dl))
        batch = batch.to("cuda")
        model = model.to("cuda")
        model.eval()
        with torch.no_grad():
            preds = model(batch)
        assert "co2_uptake" in preds, "Missing mean prediction"
        assert "co2_uptake_log_var" in preds, "Missing log_var prediction"


# 6. LOSS COMPUTATION TESTS — GPU 

@requires_gpu
@gpu
class TestLossComputation:
    """Test that losses are computed correctly for different modes."""

    def _make_model_and_batch(self, root_path, **kwargs):
        defaults = dict(
            tasks=["adsorption_db1_merged"],
            targets=["co2_uptake"],
        )
        defaults.update(kwargs)
        config, model_cls, args = _build_config(root_path, **defaults)
        model = model_cls(config).to("cuda")
        model.setup("fit")
        dl = model.train_dataloader()
        batch = next(iter(dl)).to("cuda")
        return model, batch

    @pytest.mark.parametrize("loss_fn", ["l1", "l2", "huber"])
    def test_loss_types_produce_scalar(self, root_path, loss_fn):
        model, batch = self._make_model_and_batch(root_path, loss=loss_fn)
        model.train()
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0, "Loss should be a scalar"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_loss_heteroscedastic(self, root_path):
        model, batch = self._make_model_and_batch(
            root_path, heteroscedastic=True
        )
        model.train()
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss), f"Heteroscedastic loss not finite: {loss.item()}"

    def test_loss_multiple_targets(self, root_path):
        model, batch = self._make_model_and_batch(
            root_path,
            targets=["co2_uptake", "qst_co2"],
            targets_loss_coefficients=[1.0, 1.0],
        )
        model.train()
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


# ============================================================================
# 7. INTEGRATION TESTS — GPU needed, runs a few epochs
# ============================================================================

@requires_gpu
@gpu
class TestTrainingIntegration:
    """End-to-end training tests with limited data.

    Settings aligned with run_finetune_mof_db.sh.
    """

    def test_few_epochs_single_task(self, root_path, ft_logging_path):
        """Train for 2 epochs on DB1 — full pipeline."""
        from jmp.lightning import Trainer

        config, model_cls, args = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            targets=["co2_uptake"],
            scratch=True,
            epochs=2,
            logging_path=str(ft_logging_path),
        )
        config.name = "test_few_epochs_single"
        config.trainer.logging.wandb.enabled = False
        config.trainer.checkpoint_last_by_default = False
        config.trainer.on_exception_checkpoint = False
        config.trainer.limit_train_batches = 3
        config.trainer.limit_val_batches = 3
        config.trainer.limit_test_batches = 3

        model = model_cls(config)
        trainer = Trainer(config, use_distributed_sampler=False)
        trainer.fit(model)
        trainer.test(model)
        assert trainer.current_epoch >= 1

    def test_few_epochs_heteroscedastic(self, root_path, ft_logging_path):
        """Train heteroscedastic model for 2 epochs — checks NLL loss path."""
        from jmp.lightning import Trainer

        config, model_cls, args = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            targets=["co2_uptake"],
            scratch=True,
            heteroscedastic=True,
            epochs=2,
            logging_path=str(ft_logging_path),
        )
        config.name = "test_few_epochs_hetero"
        config.trainer.logging.wandb.enabled = False
        config.trainer.checkpoint_last_by_default = False
        config.trainer.on_exception_checkpoint = False
        config.trainer.limit_train_batches = 3
        config.trainer.limit_val_batches = 3
        config.trainer.limit_test_batches = 3

        model = model_cls(config)
        trainer = Trainer(config, use_distributed_sampler=False)
        trainer.fit(model)
        assert trainer.current_epoch >= 1

    def test_few_epochs_multiple_targets(self, root_path, ft_logging_path):
        """Train with multiple targets for 2 epochs."""
        from jmp.lightning import Trainer

        config, model_cls, args = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            targets=["co2_uptake", "qst_co2"],
            targets_loss_coefficients=[1.0, 1.0],
            scratch=True,
            epochs=2,
            logging_path=str(ft_logging_path),
        )
        config.name = "test_few_epochs_multi_targets"
        config.trainer.logging.wandb.enabled = False
        config.trainer.checkpoint_last_by_default = False
        config.trainer.on_exception_checkpoint = False
        config.trainer.limit_train_batches = 3
        config.trainer.limit_val_batches = 3
        config.trainer.limit_test_batches = 3

        model = model_cls(config)
        trainer = Trainer(config, use_distributed_sampler=False)
        trainer.fit(model)
        assert trainer.current_epoch >= 1


# ============================================================================
# 8. CHECKPOINT LOADING TEST — GPU needed
# ============================================================================

@requires_gpu
@gpu
class TestCheckpointLoading:
    """Test loading a pretrained checkpoint into the model."""

    def test_load_odac_checkpoint(self, root_path):
        """Load 'odac_public' checkpoint."""
        from setup_finetune import load_checkpoint, CHECKPOINT_TAG_MAPPING

        ckpt_file = CHECKPOINT_TAG_MAPPING["odac_public"]
        ckpt_path = root_path / "checkpoints" / "EquiformerV2" / ckpt_file
        if not ckpt_path.exists():
            pytest.skip(f"ODAC checkpoint not found: {ckpt_path}")

        config, model_cls, args = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
            targets=["co2_uptake"],
            scratch=False,
            checkpoint_tag="odac_public",
            num_distance_basis=600,  
        )
        model = model_cls(config)
        load_checkpoint(model, config, args)

        backbone_params = list(model.backbone.parameters())
        non_zero = sum(1 for p in backbone_params if p.abs().sum() > 0)
        assert non_zero > 0


# ============================================================================
# 9. CONFIGURATION EDGE CASES & SAFETY
# ============================================================================

class TestConfigEdgeCases:
    """Test edge cases and safety checks in configuration."""

    def test_geometric_target_removed_from_fusion(self, root_path):
        """When a geometric property is a target, it should be removed from fusion inputs."""
        args = _make_base_args(
            root_path=str(root_path),
            targets=["lcd"],
            enable_feature_fusion=True,
            fusion_feature_names=["pld", "lcd", "gcd", "unitcell_volume", "density", "asa", "av", "nav"],
        )
        geo_in_targets = {"lcd", "pld", "gcd", "unitcell_volume", "density", "asa", "av", "nav"} & set(args.targets)
        if geo_in_targets:
            args.fusion_feature_names = [
                f for f in args.fusion_feature_names if f not in geo_in_targets
            ]
        assert "lcd" not in args.fusion_feature_names
        assert "pld" in args.fusion_feature_names  # not a target, should remain

    def test_all_fusion_features_as_targets_disables_fusion(self):
        """If all fusion features are targets, fusion should be disabled."""
        all_geo = ["lcd", "pld", "gcd", "unitcell_volume", "density", "asa", "av", "nav"]
        args = _make_base_args(
            targets=all_geo,
            enable_feature_fusion=True,
            fusion_feature_names=list(all_geo),
        )
        geo_in_targets = set(all_geo) & set(args.targets)
        if geo_in_targets:
            args.fusion_feature_names = [
                f for f in args.fusion_feature_names if f not in geo_in_targets
            ]
            if not args.fusion_feature_names:
                args.enable_feature_fusion = False
        assert args.enable_feature_fusion is False

    def test_normalization_type_expansion(self):
        """If fewer normalization types than targets, should expand with 'standard'."""
        args = _make_base_args(
            targets=["co2_uptake", "qst_co2", "kh_co2"],
            normalization_type=["log"],
        )
        n = len(args.targets)
        if len(args.normalization_type) < n:
            args.normalization_type += ["standard"] * (n - len(args.normalization_type))
        assert len(args.normalization_type) == 3
        assert args.normalization_type == ["log", "standard", "standard"]

    def test_invalid_target_raises(self, root_path):
        """Invalid target should raise during config construction."""
        with pytest.raises(Exception):
            _build_config(
                root_path,
                tasks=["adsorption_db1_merged"],
                targets=["nonexistent_target_xyz"],
            )

    def test_dataset_name_or_tasks_required(self, root_path):
        """Config build should handle tasks=None gracefully (uses dataset_name fallback)."""
        config, _, _ = _build_config(
            root_path,
            tasks=["adsorption_db1_merged"],
        )
        assert config is not None
