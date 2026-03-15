from pathlib import Path
import os
import argparse
import torch

from jmp.tasks.pretrain import PretrainConfig, PretrainModel
from jmp.lightning import Runner, Trainer
from jmp.utils.fit_scales import fit_scales_new
from jmp.modules.scaling.util import ensure_fitted
from jmp.utils.env_utils import load_env_paths

from setup_pretrain import configure_model
from datetime import timedelta

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pre-training script for JMP-L")
    
    parser.add_argument("--enable_mol_shuffle", action="store_true", help="Enable shuffling of atoms in the dataset")
    parser.add_argument("--enable_mol_shuffle_eval", action="store_true", help="Enable shuffling evaluation for autoregressive models")
    parser.add_argument("--num_perms_train", type=int, default=2, help="Number of permutations for training dataset")
    parser.add_argument("--num_perms_val", type=int, default=4, help="Number of permutations for validation dataset")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate for the optimizer")
    parser.add_argument("--num_ctx_atoms", type=float, default=0.1, help="Number of context atoms")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers")
    parser.add_argument("--train_samples_limit", type=int, default=1000000, help="Number of training samples to use")
    parser.add_argument("--val_samples_limit", type=int, default=-1, help="Number of validation samples to use")
    # parser.add_argument("--large", action="store_true", help="Load the large pre-trained checkpoint")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--load_checkpoint", type=str, help="Path to a custom checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--task", type=str,
                        required=True, help="Name of the pretraining task. Choose from: oc20, oc22, ani1x, transition1x.")
    parser.add_argument("--temperature_sampling", action="store_true", help="Use temperature sampling equal to 2")
    parser.add_argument("--logging_path", type=str, help="Path to log experiments")
    parser.add_argument("--val_interval", type=float, default=0.2, help="Validation frequency per epoch")
    parser.add_argument("--oc20_split", type=str, default="2M", help="Name of OC20 split")

    parser.add_argument("--position_norm", action="store_true", help="Apply normalization to position prediction")
    parser.add_argument("--large", action="store_true", help="Choose model size: large")
    parser.add_argument("--small", action="store_true", help="Choose model size: small")
    parser.add_argument("--medium", action="store_true", help="Choose model size: medium")
    parser.add_argument("--max_natoms", type=int, default=None, help="Filter dataset by max number of atoms")

    parser.add_argument("--no_pbc", action="store_true", help="Disable periodic boundary conditions")
    parser.add_argument("--multi_heads", action="store_true", help="Enable multiple output heads")

    parser.add_argument("--model_name", type=str, default="gemnet", help="Name of the model to use")
    parser.add_argument("--autoregressive", action="store_true", help=("Load the autoregressive"
                            "vairant of the model. Currently only supported for EquiformerV2"))
    parser.add_argument("--llama", action="store_true", help=("Load the llama variant of the model."
                            "Currently only supported for EquiformerV2"))
    parser.add_argument("--postfix", type=str, default="", help="Add a postfix to the job name")
    parser.add_argument("--resume_from_checkpoint", action="store_true",
                        help="Resume training from the last checkpoint")
    
    args = parser.parse_args()
    
    assert args.model_name in ["gemnet", "schnet", "equiformer_v2", "escaip"],( 
        f"Model name {args.model_name} not supported") 
    
    paths = load_env_paths()
    # If the user didn’t override logging_path manually, use pt_logging from YAML
    if not args.logging_path:
        args.logging_path = str(paths["pt_logging"])

    return args



def build_job_name(args, config):
    """Construct a job name based on the configuration."""
    job_name = f"PT_{args.task}_lr{args.lr}_train{args.train_samples_limit}_val{args.val_samples_limit}_ep{args.epochs}_bs{args.batch_size}"
    if args.scratch:
        job_name += f"_scratch"

    
    if args.temperature_sampling:
        job_name += "_TmpSampling"


    if args.large:
        job_name += "_large"
        
    if args.autoregressive:
        job_name += "_autoreg"

    if args.no_pbc:
        job_name += f"_noPBC"

    if args.position_norm:
        job_name += f"_posNorm"
    
    if args.multi_heads:
        job_name += f"_multiHeads"
    else:
        job_name += f"_singleHead"
    
    if args.enable_mol_shuffle:
        job_name += f"_shufflePos"
    if args.enable_mol_shuffle_eval:
        job_name += f"_withShuffleEvaluation"

    return job_name + (f"_{args.postfix}" if args.postfix else "")

def run_training(
    config: PretrainConfig, 
    model_cls: type[PretrainModel],
    args
    ):
    """Run the training process."""
    
    if args.resume_from_checkpoint:
        # model = model_cls.load_from_checkpoint(
        #     checkpoint_path=args.load_checkpoint,
        #     )
        model = model_cls.load_from_checkpoint(
            checkpoint_path=args.load_checkpoint,
            hparams=config,   # matches your __init__
            # strict=False
        )
        trainer = Trainer(config, max_epochs=args.epochs)
        print(f"Resuming training from checkpoint: {args.load_checkpoint}")
        trainer.fit(model, ckpt_path=args.load_checkpoint)
        return
    
    
    print("Creating the Model =================")
    model = model_cls(config)
    
    if args.model_name == "gemnet":
        fit_scales_new(
            config=config,
            model=model,
            backbone=lambda m: m.backbone
        )
        ensure_fitted(model)
            
    trainer = Trainer(config)
    # trainer.validate(model)
    trainer.fit(model)

def main():
    args = parse_args()
    
    config, _, init_model = configure_model(args)
    config.name = build_job_name(args, config)
    config.trainer.auto_add_trainer_finalizer = False
    
    if args.resume_from_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location="cpu")
        config = PretrainConfig.from_dict(ckpt['hyper_parameters'])
        config.trainer.auto_set_default_root_dir = False

    print("The arguments are:", args)
    print(config)

    if "SLURM_JOB_ID" in os.environ:
        os.environ["SUBMITIT_EXECUTOR"] = "slurm"
    if "RANK" not in os.environ and int(os.environ["SLURM_JOB_NUM_NODES"]) > 1:
        n_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        gpus_per_node = int(os.environ.get("NPROC_PER_NODE", 1))
        cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        timeout = timedelta(hours=4)

        # Print configuration info before submitting
        print("Submitting job with configuration:")
        print(f"  Nodes: {n_nodes}")
        print(f"  GPUs per node: {gpus_per_node}")
        print(f"  CPUs per task: {cpus_per_task}")
        print(f"  Partition: gpu")
        print(f"  Snapshot: True")
        print(f"  Timeout: {timeout}")

        runner = Runner(run_training)
        runner.submit(
            runs=[(config, init_model, args)],
            gpus=gpus_per_node,
            nodes=n_nodes,
            partition="gpu",
            cpus_per_task=cpus_per_task,
            snapshot=True,
            timeout=timeout,
        )

        return
    
    runner = Runner(run_training)
    runner([(config, init_model, args)], reset_id=False,)

if __name__ == "__main__":
    main()
