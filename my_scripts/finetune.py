import os
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
from jmp.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase
from jmp.lightning import Runner, Trainer
from jmp.modules.ema import EMA
from jmp.modules.scaling.util import ensure_fitted
from jmp.utils.env_utils import load_env_paths

def is_rank_zero():
    rank = os.environ.get("RANK")
    if rank is None:
        return True   # single-process mode
    return int(rank) == 0

from setup_finetune import (
    get_configs, load_checkpoint, configure_wandb,
    CHECKPOINT_TAG_MAPPING
)

# Set up argument parser
parser = argparse.ArgumentParser(description="Fine-tuning script for JMP-L")
parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb logging")
parser.add_argument("--dataset_name", type=str, help="Name of the dataset")

parser.add_argument("--targets", type=str, nargs='+', required=True,
                    help="List of space-separated target molecule/property for the dataset;"
                    "the first target's loss is used as primary metric")
parser.add_argument("--targets_loss_coefficients", type=float, default=1.0, nargs='+',  
                    help="List of space-separated target weights for the dataset"
                    "the order should match the order of the targets")
parser.add_argument("--norm_mean",  type=float, nargs='+', default=None, help="Mean for normalization;"
                    "the order should match the order of the targets")
parser.add_argument("--norm_std",  type=float, nargs='+', default=None, help="STD for normalization;"
                    "the order should match the order of the targets")
parser.add_argument("--normalization_type", nargs="+", choices=["log", "log1p", "standard"], default=["standard"],
                    help=("Space-seperated list of type(s) of normalization to use."
                          "Choices: 'standard' (linear), 'log' (log with abs), 'log1p' (log(1+x), stable for small values). "
                          "The order should match the order of the targets. "
                          "If not provided, defaults to 'standard' for all targets."))

parser.add_argument("--graph_scalar_reduction", nargs="+", choices=["sum", "mean"],
                    default="mean",
                    help=("Space-separated list of reduction types for graph scalar targets."
                          "For extensive target: use \'sum\'"
                          "For  intensive target: use \'mean\'"))
parser.add_argument("--node_vector_reduction", nargs="+", choices=["sum", "mean"],
                    default="mean",
                    help=("Space-separated list of reduction types for node vector targets."
                          "For extensive target: use \'sum\'"
                          "For  intensive target: use \'mean\'"))

parser.add_argument("--fold", type=int, default=0, help="Fold for Matbench dataset")
parser.add_argument("--lr", type=float, default=8.0e-5, help="Learning rate for the optimizer")
parser.add_argument("--scratch", action="store_true", help="Train from scratch")
parser.add_argument("--checkpoint_path", type=str, help="Path of finetune checkpoint to load")
parser.add_argument("--very_small_qm9", action="store_true", help="Load the small 5 layers model")
parser.add_argument("--small", action="store_true", help="Load the small pre-trained checkpoint")
parser.add_argument("--medium", action="store_true", help="Load the medium pre-trained checkpoint")
parser.add_argument("--large", action="store_true", help="Load the large pre-trained checkpoint")

parser.add_argument("--checkpoint_tag",type=str,default="jmp", 
    choices=CHECKPOINT_TAG_MAPPING.keys(),
    help="Predefined checkpoint tag. Tags with 'autoreg' trigger autoregressive logic.")

parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--batch_size", type=int, help="Training batch size", default=1)
parser.add_argument("--root_path", type=str, help="Root path containing datasets and checkpoints")
parser.add_argument("--logging_path", type=str, help="Path to log experiments")

parser.add_argument("--cutoff", type=float, default=8, help="Cutoff for the dataset")
parser.add_argument("--max_natoms", type=float, default=None, help="Filter dataset by max number of atoms")
parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for the optimizer")
parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate for regularizing the model.")
parser.add_argument("--edge_dropout", type=float, default=0.05, help="Dropout rate specifically for edge features.")
parser.add_argument("--log_predictions", action="store_true", help="Log predictions to a file")
parser.add_argument("--num_workers", type=int, default=6, help="Number of workers for data loading")
parser.add_argument("--no_pbc", action="store_true", help="Disable periodic boundary conditions")
parser.add_argument("--number_of_samples", type=int, help="Number of samples to use for training")
parser.add_argument("--loss", choices=["l1", "huber", "l2"], default="l1", help="Loss function to use")
parser.add_argument("--model_name", type=str, default="gemnet", help="Name of the model to use")
parser.add_argument("--llama", action="store_true", help=("Load the llama variant of the model."
                        "Currently only supported for EquiformerV2"))

parser.add_argument("--adabin", action="store_true", help=("Use binning strategy based on the training set"
                       "to reforumulate the regression task as a classification task."))
parser.add_argument("--adabin_dropout", type=float, default=0.5, help="Classification head dropout")
parser.add_argument("--adabin_num_classes", type=int, default=[256], nargs='+', 
                    help="Number of bins (classes) for adabin")
parser.add_argument("--adabin_class_weights", type=float, default=[1.0], nargs='+', 
                    help="Space-separated list of class weights for adabin")

parser.add_argument("--tasks", type=str, nargs='+', help=("Space-separated list of tasks"
                        "to perform; currently only 'adsorption-db-x' is supported"))
parser.add_argument("--train_samples_limit", type=float, default=-1.0, 
                    help="Limit training samples: >1 for absolute count, 0-1 for percentage (e.g., 0.2 = 20%%)")
parser.add_argument("--val_samples_limit", type=int, default=-1,
                    help="Limit the number of validation samples; useful for debugging")
parser.add_argument("--test_samples_limit", type=int, default=-1,
                    help="Limit the number of test samples; useful for debugging")
parser.add_argument("--sample_temperature", type=int, default=2,
                    help="Temperature sampling value (default is 2)")
parser.add_argument("--sample_type", choices=["temperature", "uniform"], default="temperature",
                    help="Temperature sampling type")
parser.add_argument("--max_atoms_per_batch", type=int, default=800,
                    help="Maximum number of atoms per bucket. Essentially replaces batch size.")

parser.add_argument("--atom_bucket_batch_sampler", action="store_true",
                    help="Use AtomBucketBatchSampler for training")

parser.add_argument("--uptake_conversion_mode", type=str, choices=["none", "per_cell", "per_volume"],
    default="none",
    help="Uptake unit conversion mode: 'none' (mmol/g), 'per_cell' (#CO2/cell), or 'per_volume' (#CO2/angstrom^3)")
parser.add_argument("--profile", action="store_true", help="Enable pytorch profiler")
parser.add_argument("--position_norm", action="store_true", help="Enable position normalization")
parser.add_argument("--postfix", type=str, default="", help="Add a postfix to the job name")

parser.add_argument("--compute_avg_dataset_stats", default=True, action="store_true", help="Compute avg_num_nodes on the fly from dataset (overrides backbone defaults).")
parser.add_argument("--compute_avg_dataset_stats_degree", default=True, action="store_true", help="Also compute avg_degree by sampling graphs (more expensive).")
parser.add_argument("--avg_stats_sample_size", type=int, default=10000, help="Number of samples to use when computing avg_degree.")
parser.add_argument("--roi_penalty", type=float, default=1.0, help="Multiply loss by this factor when prediction is inside the ROI. Default: 1.0 (disabled).")
parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision: 16-mixed, bf16-mixed, 32-true")
parser.add_argument("--disable_ema", action="store_true", help="Disable EMA")
parser.add_argument("--val_same_as_train", action="store_true", help="If set, only validate/test on datasets used for training.")

# RBF function and number of distance basis override
parser.add_argument("--rbf_function", type=str, default="gaussian", choices=["gaussian","coulomb_sturmian", "cs","bessel", "laplace", "hankel", "radial_mlp", "cheby"], help="RBF function for reciprocal block")
parser.add_argument("--num_distance_basis", type=int, default=None, help="Override number of distance basis functions (overrides config)")
parser.add_argument("--max_neighbors", type=int, default=32, help="Max neighbors to consider for each atom within the given cutoff.")


# Feature fusion arguments
parser.add_argument("--enable_feature_fusion", action="store_true", help="Enable adding global MOF descriptors (PLD, LCD, etc.)")
parser.add_argument("--fusion_feature_names", nargs="+", type=str,
    default=["pld", "lcd", "gcd", "unitcell_volume", "density", "asa", "av", "nav"],
    help="Which MOF features to fuse (all used simultaneously)")
parser.add_argument("--fusion_type", type=str, default="early", choices=["early", "late"], help="Fusion method: early (inject into backbone) or late (inject into head)")
parser.add_argument("--fusion_hidden_dim", type=int, default=128, help="Total hidden dimension for the fusion MLP")

GEOMETRIC_PROPERTY_NAMES = {"lcd", "pld", "gcd", "unitcell_volume", "density", "asa", "av", "nav"}

# charge embedding
parser.add_argument("--use_charge_embedding", action="store_true", default=False,
    help="Enable per-atom partial charge embedding (early fusion into backbone)")

# heteriscedastic regression 
parser.add_argument("--heteroscedastic", action="store_true", default=False,
    help="Enable heteroscedastic regression: predict mean + log-variance per target")
parser.add_argument("--hetero_nll_weight", type=float, default=1.0,
    help="Weight for Gaussian NLL loss component (default: 1.0)")
parser.add_argument("--hetero_std_weight", type=float, default=0.1,
    help="Weight for auxiliary std supervision loss (default: 0.1)")
parser.add_argument("--hetero_min_var", type=float, default=1e-6,
    help="Minimum variance to clamp for numerical stability")

parser.add_argument("--node_sh_targets", type=str, nargs='+', default=None,
    help="Per-atom SH coefficient targets (e.g., co2_sh_coeffs)")
parser.add_argument("--node_sh_lmax", type=int, default=4,
    help="Max SH degree for density prediction")
parser.add_argument("--node_sh_loss_coefficient", type=float, default=1.0,
    help="Loss coefficient for SH targets")

# extreme validation (this is a separate csv, which can be dinamically updated with new mofs coming from WP4).
# current behaviour: SET TO NONE. WE DO NOT TEST ANY ADDITIONAL EXTREME MOFs.
parser.add_argument("--extreme_val_csv", type=str, default=None,
    help="Path to CSV with ground-truth targets for extreme MOFs. "
         "Enables ExtremeValCallback when provided.")
parser.add_argument("--extreme_val_cif_dirs", type=str, nargs="+",
    default=["/ibex/project/c2261/datasets/all-adsorption/DATABASES/AI-GENERATED"],
    help="Directories containing extreme MOF CIF files.")

args = parser.parse_args()
if (not args.dataset_name and not args.tasks) or (args.dataset_name and args.tasks):
    raise ValueError("You must specify either --dataset_name or --tasks")
if (n := len(args.targets)) > (m := len(args.normalization_type)):
    if is_rank_zero():
        print(f"{args.normalization_type=}")
    args.normalization_type = args.normalization_type + (n - m) * ["standard"]


paths = load_env_paths()
# If the user didn’t override logging_path manually, use ft_logging from YAML
if not args.logging_path:
    args.logging_path = str(paths["ft_logging"])

# TODO: Support different adabin_num_classes for different targets
# if args.adabin:
#     if (n := len(args.targets)) > (m := len(args.adabin_num_classes)):
#         args.adabin_num_classes = args.adabin_num_classes + (n - m) * [256]
#     if (n := len(args.adabin_num_classes)) > (m := len(args.adabin_class_weights)):
#         args.adabin_class_weights = args.adabin_class_weights + (n - m) * [1.0]

assert args.model_name in ["gemnet", "schnet", "equiformer_v2", "escaip"],( 
    f"Model name {args.model_name} not supported") 

# if is_rank_zero():
#     print("The arguments are:", args)

def build_job_name(args, config= None):
    """Construct a job name based on the configuration."""
    # Construct the job name
    
    dataset_name_map = {
        # "adsorption-mof-db1": "DB1",
        # "adsorption-mof-db2": "DB2",
        "adsorption_db1_merged": "DB1",
        "adsorption_db2_merged": "DB2",
        "adsorption_cof_db2": "COF2",
        "adsorption_mof_db_anion": "ANION",
        "adsorption_mof_db_1_charges": "DB1_CHG",
        "adsorption_mof_db_2_charges": "DB2_CHG",
        "adsorption_mof_db_anion_charges": "ANION_CHG",
    }

    # Targets name
    if len(args.targets) == 1:
        targets_str = args.targets[0]
    else:
        targets_str = f"{len(args.targets)}targets"
        
    args.tasks = args.tasks or []
    if len(args.tasks) == 1 and " " in args.tasks[0]:
        all_tasks = args.tasks[0].split()
    else:
        all_tasks = args.tasks

    job_name_prefix = (
        dataset_name_map.get(args.dataset_name, args.dataset_name) if args.dataset_name else
        'MT[' + '+'.join([dataset_name_map.get(task, task) for task in all_tasks]) + ']' # type: ignore[assignment]
    )
    job_name = f"{job_name_prefix}_{targets_str}_ep{args.epochs}"
    
    if args.checkpoint_path:
        job_name += f"_{args.checkpoint_path}" 

    if args.dataset_name == "matbench":
        job_name += f"_fold{args.fold}" 

    if hasattr(config, 'gradient_forces') and not config.gradient_forces:
        job_name += "_direct"

    job_name += f"_LR{args.lr}"

    if args.no_pbc:
        job_name += f"_noPBC"
    
    if args.number_of_samples:
        job_name += f"_N{args.number_of_samples}"
        
    if args.scratch:
        job_name += "_scratch"
    else:
        job_name += f"_PT[{args.checkpoint_tag}]"

    if args.max_atoms_per_batch:
        print("[WARNING] Max atoms per batch is defaulting to 800 if not specified. Did you forget to set it?")
        job_name += f"_maxAtoms{args.max_atoms_per_batch}"
    elif args.batch_size:
        print("[WARNING] Batch Size is disabled by default. Please check double check.")
        job_name += f"_bs{args.batch_size}"

    if args.disable_ema:
        job_name += "_noEMA"
    
    if args.large:
        job_name += "_lrg"
    
    if args.medium:
        job_name += "_med"

    if args.small:
        job_name += "_sml"

    if args.very_small_qm9:
        job_name += "_vsml"

    if args.adabin:
        job_name += f"_adabin"

    if args.loss:
        job_name += f"_{args.loss}"

    if args.sample_type != "temperature":
        job_name += f"_balanced"

    if getattr(args, "rbf_function") is not None:
        job_name += f"_{args.rbf_function}"
    

    model_name_map = {
        "equiformer_v2": "EqV2",
    }
    if args.model_name:
        short_model_name = model_name_map.get(args.model_name, args.model_name)
        job_name += f"_{short_model_name}"

    # if args.enable_flow_matching:
        # job_name += f"_flow_matching"
    
    if args.roi_penalty and args.roi_penalty != 1.0:
        job_name += f"_roi{args.roi_penalty}"

    
    if args.enable_feature_fusion:
        n_feat = len(args.fusion_feature_names)
        fusion_type = args.fusion_type               # "early" or "late"
        job_name += f"_{n_feat}feat_{fusion_type}"

    if args.use_charge_embedding:
        job_name += "_charges"

    if getattr(args, "heteroscedastic", False):
        job_name += "_hetero"

    if args.val_same_as_train:
        job_name += f"_valAsTrain"

    if args.precision != "16-mixed":
        job_name += f"_{args.precision}"

    return job_name + (f"_{args.postfix}" if args.postfix else "")


def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    model = model_cls(config)

    if not args.scratch:
        load_checkpoint(model, config, args)

    callbacks = []
    if (ema := config.meta.get("ema")) is not None:
        callbacks.append(EMA(decay=ema))

    # We are already saving checkpoints in FinetuneModelBase
    # ckpt_dir = os.path.join(args.logging_path, "lightning_logs", config.name, "checkpoints")
    # os.makedirs(ckpt_dir, exist_ok=True)
    # callbacks.append(ModelCheckpoint(
    #     dirpath=ckpt_dir,
    #     filename="last",
    #     save_top_k=0,
    #     save_last=True,
    #     every_n_epochs=1,
    #     save_on_train_epoch_end=True,
    # ))

    # callbacks.append(ModelCheckpoint(
    #     dirpath=ckpt_dir,
    #     filename="best",
    #     monitor="val_loss",
    #     mode="min",
    #     save_top_k=1,
    #     save_last=False,
    #     save_on_train_epoch_end=False,
    # ))

    # Extreme validation callback
    if args.extreme_val_csv and "co2_uptake" in args.targets:
        from jmp.tasks.finetune.callbacks.extreme_val_callback import ExtremeValCallback
        callbacks.append(ExtremeValCallback(
            cif_dirs=args.extreme_val_cif_dirs,
            csv_path=args.extreme_val_csv,
            target="co2_uptake",
        ))

    if not args.scratch:
        ensure_fitted(model)

    trainer = Trainer(config, callbacks=callbacks, use_distributed_sampler=False)
    trainer.fit(model)
    trainer.test(model)


# If any target is a geometric property, strip it from fusion inputs (prevent data leakage)
geo_in_targets = GEOMETRIC_PROPERTY_NAMES & set(args.targets)
if geo_in_targets:
    args.fusion_feature_names = [
        f for f in args.fusion_feature_names if f not in geo_in_targets
    ]
    if args.enable_feature_fusion and not args.fusion_feature_names:
        print("[WARN] All fusion features are used as targets — disabling feature fusion.")
        args.enable_feature_fusion = False

config, init_model = get_configs(args.dataset_name, args.targets, args=args)
config.name = build_job_name(args, config)

if args.profile:
    from jmp.lightning.model.config import PyTorchProfilerConfig
    # see https://discuss.pytorch.org/t/tensor-item-takes-a-lot-of-running-time/16683
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    config.trainer.profiler = PyTorchProfilerConfig(
        dirpath="./profile_dir_2", filename="report", 
        # emit_nvtx=True, 
        export_to_chrome=True,
        
    )

if "SLURM_JOB_ID" in os.environ:
    os.environ["SUBMITIT_EXECUTOR"] = "slurm"
configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
configs.append((config, init_model))

if hasattr(args, "rbf_function") and args.rbf_function:
    config.backbone.distance_function = args.rbf_function

if hasattr(args, "num_distance_basis") and args.num_distance_basis is not None:
    config.backbone.num_distance_basis = int(args.num_distance_basis)



# for config, _ in configs:
#     assert config.backbone.scale_file, f"Scale file not set for {config.name}"

runner = Runner(run)
runner(configs)
