import torch

from pathlib import Path
import os
from jmp.utils.env_utils import load_env_paths
from jmp.configs.pretrain.jmp_l import jmp_l_pt_config_
from jmp.configs.pretrain.equiformer_v2 import (
    equiformer_v2_pt_config_, 
    autoreg_equiformer_v2_pt_config_
)
from jmp.modules.dataset.common import DatasetSampleNConfig
from jmp.tasks.pretrain import PretrainConfig
from jmp.tasks.pretrain.module import (
    PretrainModel,
    AutoregressivePretrainModel,
    NormalizationConfig,
    PositionNormalizationConfig,
    PretrainDatasetConfig,
    TaskConfig,
)


# Configure tasks based on command-line arguments
def configure_tasks(args):
    paths = load_env_paths()  # auto-detect ibex/docker and load paths.yaml
    root_path = paths["root"]
    OCP_path = paths["ocp"]


    c2261_path = root_path / "datasets"
    train_samples_limit = args.train_samples_limit
    if args.val_samples_limit > 0:
        val_samples_limit = args.val_samples_limit
    # this was set 2500 originally. I set to None, that 
    # should mean that the whole data will be used.
    # This is a reference to adsorption_db.py 
    # fn 'jmp_l_adsorption_db_multi_task_config_'
    else:
        val_samples_limit = None

    dataset_names = args.task.split(",")

    # Here we apply the ratios of temperature = 2 based on the dataset sizes reported in JMP paper
    if hasattr(args, "temperature_sampling") and args.temperature_sampling:
        temperature_limit = {
            "ani1x": int(0.0812536740566486 * train_samples_limit),
            "transition1x": int(0.18168873861227738 * train_samples_limit),
            "oc20": int(0.5745502392177768 * train_samples_limit),
            "oc22": int(0.1625073481132972 * train_samples_limit)
        }
        if sum(temperature_limit.values()) != train_samples_limit:
            temperature_limit["ani1x"] += train_samples_limit - sum(temperature_limit.values())
        
        assert sum(temperature_limit.values()) == train_samples_limit, "Temperature limit does not match train_samples_limit"


    else:
        train_samples_limit = train_samples_limit // len(dataset_names)


    is_custom_ratios = args.temperature_sampling 

    all_tasks = {
        "oc20": TaskConfig(
            name="oc20",
            train_dataset=PretrainDatasetConfig(
                src=c2261_path / f"oc20_s2ef/{args.oc20_split}/train/",
                metadata_path=c2261_path / f"oc20_s2ef/{args.oc20_split}/train_metadata.npz",
                lin_ref=c2261_path / "oc20_s2ef/2M/linref.npz",
                # Note: to have the same behavior as the EqV2 repo max_insances, uncommnet 
                # the following line and and commnet disable `sample_n` 
                # max_samples=temperature_limit["oc20"] if is_custom_ratios else train_samples_limit,
                sample_n=DatasetSampleNConfig(
                    sample_n=temperature_limit["oc20"] if is_custom_ratios else train_samples_limit,
                    seed=args.seed
                ),
                is_train=True,
                args=args
            ),
            val_dataset=PretrainDatasetConfig(
                src=c2261_path / "oc20_s2ef/all/val_id/",
                metadata_path=c2261_path / "oc20_s2ef/all/val_id_metadata.npz",
                lin_ref=c2261_path / "oc20_s2ef/2M/linref.npz",
                max_samples=val_samples_limit,
                is_train=False,
                args=args
            ),
            energy_loss_scale=1.0,
            force_loss_scale=10.0,
            normalization={
                "y": NormalizationConfig(mean=0.0, std=24.901469505465872),
                "force": NormalizationConfig(mean=0.0, std=0.5111534595489502),
                **(
                    {
                        "pos": PositionNormalizationConfig(mean=10.115131670184638, std=7.5206006550378195)
                    } if (args.autoregressive and args.position_norm) else {}
                )
            },
        ),
        "oc20relaxed": TaskConfig(
            name="oc20",
            train_dataset=PretrainDatasetConfig(
                src=OCP_path / "is2re/all/train/",
                metadata_path=c2261_path / "oc20_is2re/all/train_metadata.npz",
                lin_ref=None,
                # Note: to have the same behavior as the EqV2 repo max_insances, uncommnet 
                # the following line and and commnet disable `sample_n` 
                # max_samples=temperature_limit["oc20"] if is_custom_ratios else train_samples_limit,
                # sample_n=DatasetSampleNConfig(
                #     sample_n=temperature_limit["oc20"] if is_custom_ratios else train_samples_limit,
                #     seed=args.seed
                # ),
                is_train=True,
                args=args
            ),
            val_dataset=PretrainDatasetConfig(
                src=OCP_path / "is2re/all/val_id/",
                metadata_path=c2261_path / "oc20_is2re/all/val_id_metadata.npz",
                lin_ref=None,
                max_samples=val_samples_limit,
                is_train=False,
                args=args
            ),
            energy_loss_scale=1.0,
            force_loss_scale=10.0,
            normalization={
                **(
                    {
                        "pos": NormalizationConfig(mean=9.974598873842538, std=7.436021144547389)
                    } if (args.autoregressive and args.position_norm) else {}
                )
            },
        ),
        "oc22": TaskConfig(
            name="oc22",
            train_dataset=PretrainDatasetConfig(
                src=OCP_path / "oc22/s2ef-total/train/",
                metadata_path=c2261_path / "oc22/s2ef-total/train_metadata.npz",
                lin_ref=c2261_path / "oc22/s2ef-total/linref.npz",
                # max_samples=temperature_limit["oc22"] if is_custom_ratios else train_samples_limit,
                sample_n=DatasetSampleNConfig(
                    sample_n=temperature_limit["oc22"] if is_custom_ratios else train_samples_limit,
                    seed=args.seed
                ),
                is_train=True,
                args=args
            ),
            val_dataset=PretrainDatasetConfig(
                src=OCP_path / "oc22/s2ef-total/val_id/",
                metadata_path=c2261_path / "oc22/s2ef-total/val_id_metadata.npz",
                lin_ref=c2261_path / "oc22/s2ef-total/linref.npz",
                max_samples=val_samples_limit,
                is_train=False,
                args=args
            ),
            energy_loss_scale=1.0,
            force_loss_scale=10.0,
            normalization={
                "y": NormalizationConfig(mean=0.0, std=25.229595396538468),
                "force": NormalizationConfig(mean=0.0, std=0.25678861141204834),
                **(
                    {
                        "pos": PositionNormalizationConfig(mean=10.327511465240537, std=8.513327060327757)
                    } if (args.autoregressive and args.position_norm) else {}
                )
            },
        ),
        "ani1x": TaskConfig(
            name="ani1x",
            train_dataset=PretrainDatasetConfig(
                src=c2261_path / "ani1x/lmdb/train/",
                metadata_path=c2261_path / "ani1x/lmdb/train/metadata.npz",
                lin_ref=c2261_path / "ani1x/linref.npz",
                # max_samples=temperature_limit["ani1x"] if is_custom_ratios else train_samples_limit,
                sample_n=DatasetSampleNConfig(
                    sample_n=temperature_limit["ani1x"] if is_custom_ratios else train_samples_limit,
                    seed=args.seed
                ),
                is_train=True,
                args=args
            ),
            val_dataset=PretrainDatasetConfig(
                src=c2261_path / "ani1x/lmdb/val/",
                metadata_path=c2261_path / "ani1x/lmdb/val/metadata.npz",
                lin_ref=c2261_path / "ani1x/linref.npz",
                max_samples=val_samples_limit,
                is_train=False,
                args=args
            ),
            energy_loss_scale=1.0,
            force_loss_scale=10.0,
            cutoff= 8.0,
            normalization={
                "y": NormalizationConfig(mean=0.0, std=2.8700712783472118),
                "force": NormalizationConfig(mean=0.0, std=2.131422996520996),
                **(
                    {
                        "pos": PositionNormalizationConfig(mean=0.019459555430128644, std=1.431408124423159)
                    } if (args.autoregressive and args.position_norm) else {}
                )
            },
        ),
        "transition1x": TaskConfig(
            name="transition1x",
            train_dataset=PretrainDatasetConfig(
                src=c2261_path / "transition1x/lmdb/train/",
                metadata_path=c2261_path / "transition1x/lmdb/train/metadata.npz",
                lin_ref=c2261_path / "transition1x/linref.npz",
                # max_samples=temperature_limit["transition1x"] if is_custom_ratios else train_samples_limit,
                sample_n=DatasetSampleNConfig(
                    sample_n=temperature_limit["transition1x"] if is_custom_ratios else train_samples_limit,
                    seed=args.seed
                ),
                is_train=True,
                args=args
            ),
            val_dataset=PretrainDatasetConfig(
                src=c2261_path / "transition1x/lmdb/val/",
                metadata_path=c2261_path / "transition1x/lmdb/val/metadata.npz",
                lin_ref=c2261_path / "transition1x/linref.npz",
                max_samples=val_samples_limit,
                is_train=False,
                args=args
            ),
            energy_loss_scale=1.0,
            force_loss_scale=10.0,
            cutoff= 8.0,
            normalization={
                "y": NormalizationConfig(mean=0.0, std=1.787466168382901),
                "force": NormalizationConfig(mean=0.0, std=0.3591422140598297),
                **(
                    {
                        "pos": PositionNormalizationConfig(mean=-0.00035213493961080543, std=1.242247780280691)
                    } if (args.autoregressive and args.position_norm) else {}
                )
            },
        ),
        "odac": TaskConfig(
            name="odac",
            train_dataset=PretrainDatasetConfig(
                src=c2261_path / "odac/s2ef/train/",
                metadata_path=c2261_path / "odac/s2ef/train_metadata.npz",
                lin_ref=c2261_path / "odac/s2ef/linref.npz",
                sample_n=DatasetSampleNConfig(
                    sample_n=train_samples_limit,
                    seed=args.seed,
                    max_natoms=args.max_natoms
                ),
                is_train=True,
                args=args
            ),
            val_dataset=PretrainDatasetConfig(
                src=c2261_path / "odac/s2ef/val/",
                metadata_path=c2261_path / "odac/s2ef/val_metadata.npz",
                lin_ref=c2261_path / "odac/s2ef/linref.npz",
                max_samples=val_samples_limit,
                is_train=False,
                args=args
            ),
            energy_loss_scale=1.0,
            force_loss_scale=10.0,
            normalization={
                "y": NormalizationConfig(mean=0.0, std=0.7627581438363593),
                "force": NormalizationConfig(mean=0.0, std=0.04262716323137283),
                **(
                    {
                        "pos": PositionNormalizationConfig(mean=8.340232861715359, std=6.98569472555122)
                    } if (args.autoregressive and args.position_norm) else {}
                )
            },
        ),
    }

    # Change the default max_neighbors of 30 to 20 to match
    # EquiformerV2 orginial repo
    for _, config in all_tasks.items():
        if args.model_name == "equiformer_v2" and args.autoregressive:
            config.max_neighbors = 20

    print(all_tasks)
    # Filter tasks based on dataset_names
    tasks = [all_tasks[dataset_name] for dataset_name in dataset_names if dataset_name in all_tasks]
    return tasks, root_path



def configure_model(args):
    """Set up the model configuration based on command-line arguments."""
    config = PretrainConfig.draft()
    config.args = args
    init_model = PretrainModel
    if args.model_name == "gemnet":
        jmp_l_pt_config_(config, args)
    
    elif args.model_name == "equiformer_v2" and not args.autoregressive:
        equiformer_v2_pt_config_(config, args)
        
    elif args.model_name == "equiformer_v2" and args.autoregressive:
        autoreg_equiformer_v2_pt_config_(config, args)
        if args.multi_heads:
            config.multi_heads = True
        
        init_model = AutoregressivePretrainModel
        

        
    config.tasks, root_path = configure_tasks(args)
    config = config.finalize()
    configure_wandb(config, args)
    configure_validation_and_scheduler(config, args)
    
    return config, root_path, init_model


def configure_wandb(config, args):
    """Configure WandB settings."""
    if args.enable_wandb:
        config.project = "aramco_dac_pretrain"
        config.entity = "theme4"
    else:
        config.trainer.logging.wandb.enabled = False


def configure_validation_and_scheduler(config, args):
    """Set validation and scheduler parameters."""
    # config.trainer.val_check_interval = 0.0002  # Run validation every 0.02% of the epoch (almost every hour)
    # config.lr_scheduler.max_epochs = None
    # config.lr_scheduler.max_steps = 800000

    # config.trainer.val_check_interval = 0.25  # Run validation every 25% of the epoch
    # config.trainer.val_check_interval = None  # this was added for debugging
    # config.lr_scheduler.max_epochs = None # this is set in the respective model config


    # Note: the max_steps calculation is being handled in configuring the lr scheduler in 
    # `tasks/pretrain/module.py`. For consistency with the original EqV2 repo, the effective
    # batch size is not handled, instead it is computed as follows:
    # max_steps = ceil(ceil(batch_size / num_gpus) / accumulated_grad_batches) * epochs
      
    # num_gpus = torch.cuda.device_count()
    # effective_batch_size = args.batch_size * num_gpus
    # the `train_samples_limit` is 
    # max_steps = (args.train_samples_limit // effective_batch_size) * args.epochs
    # config.lr_scheduler.max_steps = max_steps
