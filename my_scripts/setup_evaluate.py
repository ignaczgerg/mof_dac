import os
from pathlib import Path
import torch
from pydantic import ValidationError

from jmp.configs.finetune.jmp_l import jmp_l_ft_config_
from jmp.configs.finetune.schnet import schnet_ft_config_
from jmp.configs.finetune.escaip import escaip_small_ft_config_
from jmp.configs.finetune.equiformer_v2 import (
    equiformer_v2_ft_config_, 
    autoreg_llama_equiformer_v2_ft_config_,
)

from jmp.configs.finetune.rmd17 import jmp_l_rmd17_config_
from jmp.configs.finetune.qm9 import (
    jmp_l_qm9_config_, 
    escaip_small_qm9_config_,
    equiformer_v2_qm9_config_,
)
from jmp.configs.finetune.md22 import jmp_l_md22_config_
from jmp.configs.finetune.spice import jmp_l_spice_config_
from jmp.configs.finetune.matbench import jmp_l_matbench_config_
from jmp.configs.finetune.qmof import jmp_l_qmof_config_
from jmp.configs.finetune.adsorption_db import (
    jmp_l_adsorption_db_config_, 
    escaip_small_adsorption_db_config_, 
    equiformer_v2_adsorption_db_config_,
    jmp_l_adsorption_db_config_adabin_,
    jmp_l_adsorption_db_multi_task_config_,
    equiformer_v2_adsorption_db_multitask_config_,
)

from jmp.tasks.finetune.rmd17 import RMD17Config, RMD17Model
from jmp.tasks.finetune.qm9 import QM9Config, QM9Model
from jmp.tasks.finetune.md22 import MD22Config, MD22Model
from jmp.tasks.finetune.spice import SPICEConfig, SPICEModel
from jmp.tasks.finetune.matbench import MatbenchConfig, MatbenchModel
from jmp.tasks.finetune.qmof import QMOFConfig, QMOFModel
from jmp.tasks.finetune.adsorption_db import AdsorptionDbConfig, AdsorptionDbModel

from jmp.tasks.finetune import dataset_config as DC



from jmp.utils.finetune_state_dict import (
    filter_state_dict,
    retreive_state_dict_for_finetuning,
    load_equiformer_ema_weights
)

def get_configs(dataset_name: str, targets: list[str], args):
    
    if args.checkpoint_path:
        print("Loading custom (finetuned) checkpoint =================")
        ckpt_path = Path(args.checkpoint_path)

    elif args.model_name == "gemnet":
        ckpt_path = Path(args.root_path) / "checkpoints/JMP/jmp-s.pt"
        if args.large:
            ckpt_path = Path(args.root_path) / "checkpoints/JMP/jmp-l.pt"

    elif args.model_name == "equiformer_v2":
        if args.medium:
            checkpoint_name = "eq2_83M_2M.pt"
        elif args.large:
            checkpoint_name = "eq2_153M_ec4_allmd.pt"
        else:
            if args.oc20_small_checkpoint:
                checkpoint_name = "eq2_31M_ec4_allmd.pt"
            elif args.odac_small_checkpoint:
                checkpoint_name = "eqv2_31M_odac.pt"
            elif args.autoregressive and not args.llama:
                # checkpoint_name = "final_autoreg.pt"
                checkpoint_name = "autoregressive/OC20_All/autoreg_onlyPositions_epoch@0.124_OC20All.pt"
            elif args.autoregressive and args.llama:
                checkpoint_name = "autoregLlamaEquiformerV2.pt"
            else:
                checkpoint_name = "eq2_31M_jmp_epoch_1_step_260k_oarks1e9.ckpt"        
        
        ckpt_path = Path(args.root_path) / f"checkpoints/EquiformerV2/{checkpoint_name}"
        
    else: 
        ckpt_path = None

    dataset_name = dataset_name.lower()
    base_path = Path(args.root_path) / f"datasets/{dataset_name}/"
    print("======================")
    print("[WARNING]:", base_path)
    print("======================")
    if getattr(args, "multitask", False):
        config = AdsorptionDbConfig.draft()
        if args.model_name == "gemnet":
            jmp_l_ft_config_(config, ckpt_path, args=args)
            jmp_l_adsorption_db_multi_task_config_(config, targets, args=args)
        elif args.model_name == "equiformer_v2":
            equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_adsorption_db_multitask_config_(config, targets, args=args)
        else:
            raise ValueError("Multitask currently supported for model_name in {'gemnet', 'equiformer_v2'}.")

        init_model = AdsorptionDbModel
        config.args = args
        try:
            config = config.finalize()
        except ValidationError as e:
            missing_fields = [er['loc'][0] for er in e.errors() if er['type'] == 'missing']
            print("Missing fields:", missing_fields)
        configure_wandb(config, args)
        print(config)
        return config, init_model


    # if dataset_name == "adsorption-db1":
    #     config = AdsorptionDbConfig.draft()
    #     jmp_l_ft_config_(config, ckpt_path, args=args)  
    #     jmp_l_adsorption_db_config_(config, targets, base_path, args)
    #     init_model = AdsorptionDbModel

    # elif dataset_name == "adsorption-db2":
    #     config = AdsorptionDbConfig.draft()
    #     jmp_l_ft_config_(config, ckpt_path, args=args)  
    #     jmp_l_adsorption_db_config_(config, targets, base_path)
    #     init_model = AdsorptionDbModel
        
    # elif dataset_name in ["adsorption-mof-db1", "adsorption-mof-db2"]:
    elif dataset_name in ["adsorption_db1_merged", "adsorption_db2_merged"]:
        config = AdsorptionDbConfig.draft()
        # config.meta["csv_path"] = (
        #     base_path / "raw" / "MOF-DB-1.0_DATABASE_28012025.csv"
        #     if dataset_name == "adsorption-mof-db1" else
        #     base_path / "raw" / "MOF_DB_2_0_18042025.csv"
        #     )
        if args.model_name == "gemnet" and args.adabin:
            jmp_l_ft_config_(config, ckpt_path, args=args)
            jmp_l_adsorption_db_config_adabin_(config, targets, base_path, args)
        elif args.model_name == "gemnet":
            jmp_l_ft_config_(config, ckpt_path, args=args)
            jmp_l_adsorption_db_config_(config, targets, base_path, args)
        elif args.model_name == "schnet":
            schnet_ft_config_(config, ckpt_path, args=args)
            jmp_l_adsorption_db_config_(config, targets, base_path, args)
        elif args.model_name == "escaip":
            escaip_small_ft_config_(config, ckpt_path, args=args)
            escaip_small_adsorption_db_config_(config, targets, base_path, args)
        elif args.model_name == "equiformer_v2" and not args.llama: 
            equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_adsorption_db_config_(config, targets, base_path, args)
        elif args.model_name == "equiformer_v2" and args.llama:
            autoreg_llama_equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_adsorption_db_config_(config, targets, base_path, args)
        
        if Path(args.path_to_cifs).exists():
            config.test_dataset = DC.adsorption_db_config(Path(args.path_to_cifs), "test", args=args)
        init_model = AdsorptionDbModel
        config.tasks = []
        config.train_tasks = []
        config.use_balanced_batch_sampler = False
        config.trainer.use_distributed_sampler = False

    elif dataset_name == "adsorption-cof-db1":
        assert False, "Not implemented yet"
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config.args = args
    

    # check which fields are missing
    from pydantic import BaseModel, ValidationError
    try:
        config = config.finalize()
    except ValidationError as e:
        missing_fields = [error['loc'][0] for error in e.errors() if error['type'] == 'missing']
        print("Missing fields:", missing_fields)

    # breakpoint()

    configure_wandb(config, args)
    print(config)
    return config, init_model

def expand_embedding_state_dict(state_dict, new_embedding_shape):
    """Expand the embeddings in the state dictionary to match the new number of elements for specific keys."""
    keys_to_expand = [
        "backbone.sphere_embedding.weight",
        "backbone.edge_degree_embedding.source_embedding.weight",
        "backbone.edge_degree_embedding.target_embedding.weight",
    ]

    num_new_elements, embedding_dim = new_embedding_shape
    
    for i in range(8):  # Adjust based on the max number of blocks
        keys_to_expand.extend([
            f"backbone.blocks.{i}.ga.source_embedding.weight",
            f"backbone.blocks.{i}.ga.target_embedding.weight",
        ])

    for key in keys_to_expand:
        if key in state_dict and state_dict[key].shape[0] != num_new_elements:
            old_weights = state_dict[key]
            new_weights = torch.randn((num_new_elements, embedding_dim), device=old_weights.device).uniform_(-1, 1)
            new_weights[:old_weights.shape[0], :] = old_weights
            state_dict[key] = new_weights
    
    return state_dict


def save_md_state_dict(state_dict: dict[str, torch.Tensor], name: str = "model_state"):
    """Print the state dictionary in a tabular format."""
    import pandas as pd

    df = pd.DataFrame([
        {"param": name, "shape": tuple(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in state_dict.items()
    ])

    md = df.to_markdown(index=False)
    with open(f"{name}.md", "w") as f:
        f.write(md)

def load_checkpoint(model, config, args):
    # Here, we load our fine-tuned model on the same task (including the head)

    
    # Here, we load Meta-AI original foundation model (without the heads)
    # else:

    if (ckpt_path := config.meta.get("ckpt_path")) is None:
        raise ValueError("No checkpoint path provided")
    print("Loading pretraining checkpoint =================", ckpt_path)

    if args.checkpoint_path:
        print("Loading custom checkpoint =================")
        raw = torch.load(ckpt_path, map_location="cpu")
        state_dict = raw.get("state_dict", raw)

        DROP_PREFIXES = ("task_steps.",)
        state_dict = {k: v for k, v in state_dict.items()
                    if not k.startswith(DROP_PREFIXES)}

        # 1. Load normal model weights
        incompatible = model.load_state_dict(state_dict, strict=False)
        print("Loaded with strict=False")
        print("  missing:", len(incompatible.missing_keys))
        print("  unexpected:", len(incompatible.unexpected_keys))

        # 2. Load EMA into the model
        if args.model_name == "equiformer_v2" and config.meta.get("ema_backbone", False):
            load_equiformer_ema_weights(raw, model)
        return

    

    # Load pre-trained model (requiring finetuning)
    if args.model_name == "gemnet":
        state_dict = retreive_state_dict_for_finetuning(
            ckpt_path, load_emas=config.meta.get("ema_backbone", False)
        )
        embedding = filter_state_dict(state_dict, "embedding.atom_embedding.")
        backbone = filter_state_dict(state_dict, "backbone.")
        # save_md_state_dict(state_dict, "gemnet_model_state_before_filter")
        model.load_backbone_state_dict(model.backbone, model.embedding,
                                       backbone=backbone, embedding=embedding, strict=True)

    elif args.model_name == "equiformer_v2":
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
    
        # Adjust key names for compatibility
        state_dict = {key.replace("module.module", "backbone"): value for key, value in state_dict.items()}
        state_dict = {
            key.replace("backbone.energy_block", "output.energy_heads.0.energy_block")
            .replace("backbone.force_block", "output.force_heads.0.force_block"): value
            for key, value in state_dict.items()
        }

        # save_md_state_dict(state_dict, "backbone_state_before_autoreg_filter")
        # breakpoint()

        if args.autoregressive:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("backbone.coord_head.")}
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("backbone.atomic_num_head.")}

        # save_md_state_dict(state_dict, "backbone_state_after_autoreg_filter")
        # breakpoint()

        if "mof" in args.dataset_name:
            new_embedding_shape = model.backbone.sphere_embedding.weight.shape
            state_dict = expand_embedding_state_dict(state_dict, new_embedding_shape)

        save_md_state_dict(state_dict, "autoreg_llama_backbone_state_after_embedding_expand")
        backbone_state = filter_state_dict(state_dict, "backbone.")
        model.backbone.load_state_dict(backbone_state)

        # Load EMA into the model
        if config.meta.get("ema_backbone", False):
            load_equiformer_ema_weights(ckpt, model)

def configure_wandb(config, args):
    """Configure WandB settings."""
    if args.enable_wandb:
        config.project = "new-dac-project"
    else:
        config.trainer.logging.wandb.enabled = False
