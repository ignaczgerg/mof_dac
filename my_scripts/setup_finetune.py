import os
from pathlib import Path
import torch
from pydantic import BaseModel, ValidationError

from jmp.utils.env_utils import load_env_paths
from jmp.configs.finetune.jmp_l import jmp_l_ft_config_
from jmp.configs.finetune.schnet import schnet_ft_config_
from jmp.configs.finetune.escaip import escaip_small_ft_config_
from jmp.configs.finetune.equiformer_v2 import (
    equiformer_v2_ft_config_, 
    autoreg_llama_equiformer_v2_ft_config_,
)

from jmp.configs.finetune.rmd17 import (
    jmp_l_rmd17_config_,
    equiformer_v2_rmd17_config_,
)
from jmp.configs.finetune.qm9 import (
    jmp_l_qm9_config_, 
    escaip_small_qm9_config_,
    equiformer_v2_qm9_config_,
)
from jmp.configs.finetune.md22 import (
    jmp_l_md22_config_, 
    equiformer_v2_md22_config_,
)
from jmp.configs.finetune.spice import jmp_l_spice_config_
from jmp.configs.finetune.matbench import jmp_l_matbench_config_
from jmp.configs.finetune.qmof import (
    jmp_l_qmof_config_, 
    equiformer_v2_qmof_config_,
)
from jmp.configs.finetune.adsorption_db import (
    jmp_l_adsorption_db_config_, 
    escaip_small_adsorption_db_config_, 
    equiformer_v2_adsorption_db_config_,
    jmp_l_adsorption_db_config_adabin_, 
    equiformer_v2_adsorption_db_config_adabin_, 
    autoreg_llama_equiformer_v2_adsorption_db_config_adabin_, 
    jmp_l_adsorption_db_multi_task_config_,
    equiformer_v2_adsorption_db_multitask_config_,
)

from jmp.configs.finetune.hmof import equiformer_v2_hmof_config_
from jmp.configs.finetune.obelix import equiformer_v2_obelix_config_

from jmp.tasks.finetune.rmd17 import RMD17Config, RMD17Model
from jmp.tasks.finetune.qm9 import QM9Config, QM9Model
from jmp.tasks.finetune.md22 import MD22Config, MD22Model
from jmp.tasks.finetune.spice import SPICEConfig, SPICEModel
from jmp.tasks.finetune.matbench import MatbenchConfig, MatbenchModel
from jmp.tasks.finetune.qmof import QMOFConfig, QMOFModel
from jmp.tasks.finetune.adsorption_db import AdsorptionDbConfig, AdsorptionDbModel
from jmp.tasks.finetune.hmof import HMofConfig, HMofModel
from jmp.tasks.finetune.obelix import ObelixConfig, ObelixModel


from jmp.utils.finetune_state_dict import (
    filter_state_dict,
    retreive_state_dict_for_finetuning,
    load_equiformer_ema_weights
)

CHECKPOINT_TAG_MAPPING = {
    "jmp": 
        "eq2_31M_jmp_epoch_1_step_260k_oarks1e9.ckpt",
    "oc20_public": 
        "eq2_31M_ec4_allmd.pt",
    "odac_public": 
        "eqv2_31M_odac.pt",
    "odac_large_public":
        "eqv2_153M_odac.pt",
    "oc20_supervised": 
        "eq2_31M_oc20_1M_4epochs_supervised.ckpt",
    "oc20_2M_autoreg_onlyPos":
        "autoregressive/OC20_2M/autoreg_onlyPositions_epoch@3_OC20_2M.pt", 
    "oc20_All_autoreg_onlyPos":
        "autoregressive/OC20_All/autoreg_onlyPositions_epoch@0.124_OC20All.pt",     
    "ani1x_autoreg": 
        "eq2_31M_ani1x_1M_4epochs_autoreg.ckpt",
    "oc20_autoreg_old": 
        "eq2_31M_oc20_1M_4epochs_autoreg.ckpt",
    "odac25_full_autoreg":
        "runpod_ckpts/2025-10-19-10-16-32_ODAC25/best_checkpoint.pt",
    "odac25_flow_matching":
        "FlowMatch_vf/ODAC25_2M/N@8_L@4_M@2_31M/bs@512_lr@4e-4_wd@1e-3_epochs@3_warmup-epochs@0.01_g@8x8_position_coefficient@1_atm_num_coefficient@0.1/checkpoints/2025-11-02-20-40-16/best_checkpoint_flow.pt",
    "odac25_flow_matching_reg1":
        "FlowMatch_vf/ODAC25_2M/N@8_L@4_M@2_31M/bs@512_lr@4e-4_wd@1e-3_epochs@3_warmup-epochs@0.01_g@8x8_position_coefficient@1_atm_num_coefficient@0.1/checkpoints/2025-11-02-20-38-08/best_checkpoint_flow.pt",
    "odac25_flow_matching_reg1_vf":
        "FlowMatch_vf/ODAC25_2M/N@8_L@4_M@2_31M/bs@512_lr@4e-4_wd@1e-3_epochs@3_warmup-epochs@0.01_g@8x8_position_coefficient@1_atm_num_coefficient@0.1/checkpoints/2025-11-03-23-39-28/best_checkpoint_flow.pt",
    "odac25_flow_matching_reg1_vRunPod":
        "FlowMatch_vf/ODAC25_2M/runpod/checkpoint.pt",
    "odac25_flow_matching_reg1_vRunPod_v2":
        "FlowMatch_vf/ODAC25_2M/runpod/checkpoint_v2.pt",
    "odac25_flow_matching_reg1_vRunPod_best@2epochs":
        "FlowMatch_vf/ODAC25_2M/runpod/best_checkpoint@2epochs.pt",
    "oc20_andres_autoreg": 
        "autoregressive/OC20_2M/N@8_L@4_M@2_31M/bs@512_lr@4e-4_wd@1e-3_epochs@3_warmup-epochs@0.01_g@8x8_position_coefficient@1_atm_num_coefficient@1_num_ctx_atoms@0.1_atm_weighted_acc/checkpoints/2025-05-23-14-16-16/best_checkpoint.pt",
    "oc20_autoreg_norm": 
        "eq2_31M_oc20_1M_4epochs_autoreg_norm.ckpt",
    "oc20_autoreg_noPBC_norm": 
        "eq2_31M_oc20_1M_4epochs_autoreg_noPBC_norm.ckpt",
    "oc20_autoreg_2M_6E": 
        "eq2_31M_oc20_2M_6epochs_autoreg.ckpt",
    "oc20_autoreg_9M_1E": 
        "eq2_31M_oc20_9M_1epochs_autoreg.ckpt",
    "odac_autoreg": 
        "eq2_31M_odac_1M_4epochs_autoreg_last.ckpt",
    "odac_autoreg_noPBC_norm": 
        "eq2_31M_odac_1M_1epochs_autoreg_noPBC_norm.ckpt",
    "jmp_autoreg": 
        "eq2_31M_jmp_1M_4epochs_autoreg.ckpt",
    "oc20_llama_autoreg": 
        "autoregLlamaEquiformerV2.pt",
    "oc20_2M_autoreg_epoch_03": 
        "autoreg_acp3fyu8_epoch_03.ckpt",
    "oc20_2M_autoreg_epoch_07": 
        "autoreg_acp3fyu8_epoch_07.ckpt",
    "oc20_2M_autoreg_epoch_11": 
        "autoreg_acp3fyu8_epoch_11.ckpt",
    "oc20_2M_autoreg_epoch_14": 
        "autoreg_acp3fyu8_epoch_14.ckpt",
    "oc20_2M_autoreg_epoch_29": 
        "autoreg_xhlchyke_epoch_29.ckpt",
    "oc20_All_autoreg_epoch_0.7": 
        "autoreg_za34pmgl_step_24500.ckpt",
    "jmp_autoreg_epoch_04": 
        "autoreg_duzydts4_epoch_04.ckpt",
    "ani1x_2M_autoreg_epoch_07":
        "autoreg_bpwhvse8_epoch_07.ckpt",
    "ani1x_2M_autoreg_epoch_14":
        "autoreg_bpwhvse8_epoch_14.ckpt",
    "ani1x_2M_supervised_epoch_07":
        "supervised_01f7sg6f_epoch_07.ckpt",
    "ani1x_2M_supervised_epoch_14":
        "supervised_01f7sg6f_epoch_14.ckpt",
    "all_datasets_8M_autoreg_epoch_00_posNorm":
        "autoreg_i3u4tkor_epoch_00.ckpt",
    "all_datasets_8M_autoreg_epoch_01_posNorm":
        "autoreg_i3u4tkor_epoch_01.ckpt",
    "all_datasets_8M_autoreg_epoch_00":
        "autoreg_wxehdorp_epoch_00.ckpt",
    "all_datasets_8M_autoreg_singleHead_epoch_09":
        "autoreg_wxehdorp_epoch_9.ckpt", # (last epoch) wandb run: https://wandb.ai/theme4/aramco_dac_pretrain/runs/wxehdorp 
    "all_datasets_8M_autoreg_posNorm_singleHead_epoch_09":
        "autoreg_i3u4tkor_epoch_9.ckpt", # (last epoch) wandb run: https://wandb.ai/theme4/aramco_dac_pretrain/runs/i3u4tkor
    "all_datasets_2M_autoreg_posNorm_singleHead_epoch_29":
        "autoreg_vpy71iv8_epoch_29.ckpt", # (last epoch) wandb run: https://wandb.ai/theme4/aramco_dac_pretrain/runs/vpy71iv8
    "all_datasets_2M_autoreg_posNorm_singleHead_epoch_20":
        "autoreg_vpy71iv8_epoch_20.ckpt", # (last epoch) wandb run: https://wandb.ai/theme4/aramco_dac_pretrain/runs/vpy71iv8
    "all_datasets_2M_autoreg_posNorm_singleHead_epoch_15":
        "autoreg_vpy71iv8_epoch_15.ckpt", # (last epoch) wandb run: https://wandb.ai/theme4/aramco_dac_pretrain/runs/vpy71iv8
    "all_datasets_2M_autoreg_posNorm_singleHead_epoch_10":
        "autoreg_vpy71iv8_epoch_10.ckpt", # (last epoch) wandb run: https://wandb.ai/theme4/aramco_dac_pretrain/runs/vpy71iv8
    "all_datasets_2M_autoreg_singleHead_epoch_29":
        "autoreg_bi0agq7b_epoch_29.ckpt", # (last epoch) wandb run: https://wandb.ai/theme4/aramco_dac_pretrain/runs/bi0agq7b
    "all_datasets_2M_autoreg_TmpSampling_posNorm_singleHead_epoch_29":
        "autoreg_lmrl1n73_epoch_29.ckpt", # (last epoch) wandb run: https://wandb.ai/theme4/aramco_dac_pretrain/runs/lmrl1n73
    "all_datasets_2M_autoreg_TmpSampling_singleHead_epoch_29":
        "autoreg_v71a15z9_epoch_29.ckpt", # (last epoch) wandb run: https://wandb.ai/theme4/aramco_dac_pretrain/runs/v71a15z9
    "oc20_permutation_invariant":
        "autoreg_5zzb58or_oc20_permutation_invariant_last.ckpt",
    "all_dataset_u5np46cy_tmp_epoch_14":
        "autoreg_all_dataset_u5np46cy_tmp_epoch_14.ckpt",
    "oc20_autoreg_single_vector_head":
        "autoreg_k5b0bdt1_epoch_14.ckpt",
    "odac_autoreg_permutation_invariant":
        "autoreg_odac_qopclwyz_epoch_1.ckpt", # checkpoint qopclwyz on wandb
    "flowmatching_20260114":
        "FlowMatch_vf/ODAC25_2M/lastest_ckpt/best_checkpoint_flow.pt",
    "flowmatching_20260126":
        "FlowMatch_vf/ODAC25_2M/N@8_L@4_M@2_31M/bs@512_lr@4e-4_wd@1e-3_epochs@3_warmup-epochs@0.01_g@8x8_position_coefficient@1_atm_num_coefficient@0.1_final/checkpoints/2026-01-22-21-44-16/best_checkpoint_flow.pt",
}

def is_rank_zero():
    rank = os.environ.get("RANK")
    if rank is None:
        return True   # single-process mode
    return int(rank) == 0

def set_root_path(args):
    if getattr(args, "root_path", None):
        return
    paths = load_env_paths()  # auto-detect ibex/docker and load paths.yaml
    args.root_path = paths["root"]


def get_configs(dataset_name: str, targets: list[str], args):
    set_root_path(args)
    
    if args.checkpoint_path:
        print("Loading custom (finetuned) checkpoint =================")
        ckpt_path = Path(args.checkpoint_path)

    elif args.model_name == "gemnet":
        ckpt_path = Path(args.root_path) / "checkpoints/JMP/jmp-s.pt"
        if args.large:
            ckpt_path = Path(args.root_path) / "checkpoints/JMP/jmp-l.pt"

    elif args.model_name == "equiformer_v2":
        if args.medium:
            ckpt_name = "eq2_83M_2M.pt"
            ckpt_path = Path(args.root_path) / f"checkpoints/EquiformerV2/{ckpt_name}"
        # elif args.large:
        #     ckpt_name = "eq2_153M_ec4_allmd.pt"
        else:
            if ckpt_name := CHECKPOINT_TAG_MAPPING.get(args.checkpoint_tag, None):
                ckpt_path = Path(args.root_path) / f"checkpoints/EquiformerV2/{ckpt_name}"
            else:
                raise ValueError(f"Unknown checkpoint_tag: '{args.checkpoint_tag}'")

    else:
        ckpt_path = None

    if not dataset_name and getattr(args, "dataset_name", None):
        dataset_name = args.dataset_name
    dataset_name = dataset_name.lower() if dataset_name else None 
    base_path = Path(args.root_path) / f"datasets/{dataset_name}/"
    
    if dataset_name == "rmd17":
        config = RMD17Config.draft()
        if args.model_name == "gemnet":
            jmp_l_ft_config_(config, ckpt_path, args=args)
            jmp_l_rmd17_config_(config, targets, base_path, args=args)
        elif args.model_name == "equiformer_v2":
            equiformer_v2_ft_config_(
                config, ckpt_path, args=args, disable_force_output_heads=False)
            equiformer_v2_rmd17_config_(config, targets, base_path, args=args)
        else:
            raise ValueError(f"Unsupported model for rMD17: {args.model_name}")
        init_model = RMD17Model

    elif dataset_name == "qm9":

        config = QM9Config.draft()
        if args.model_name == "gemnet":
            jmp_l_ft_config_(config, ckpt_path, args=args)  
            jmp_l_qm9_config_(config, targets, base_path, args=args)
        elif args.model_name == "equiformer_v2":
            equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_qm9_config_(config, targets, base_path, args=args)
        elif args.model_name == "escaip":
            escaip_small_ft_config_(config, ckpt_path, args=args)
            escaip_small_qm9_config_(config, targets, base_path, args=args)
        elif args.model_name == "schnet":
            raise NotImplementedError("Schnet config for QM9 not implemented yet")
        init_model = QM9Model

    elif dataset_name == "md22":
        config = MD22Config.draft()
        if args.model_name == "gemnet":
            jmp_l_ft_config_(config, ckpt_path, args=args)  
            jmp_l_md22_config_(config, targets, base_path, args=args)
        elif args.model_name == "equiformer_v2":
            equiformer_v2_ft_config_(
                config, ckpt_path, args=args, disable_force_output_heads=False)
            equiformer_v2_md22_config_(config, targets, base_path, args=args)
        else:
            raise ValueError(f"Unsupported model for MD22: {args.model_name}")
        init_model = MD22Model

    elif dataset_name == "spice":
        config = SPICEConfig.draft()
        jmp_l_ft_config_(config, ckpt_path, args=args)  
        jmp_l_spice_config_(config, targets, base_path, args=args)
        init_model = SPICEModel

    elif dataset_name == "matbench":
        config = MatbenchConfig.draft()
        jmp_l_ft_config_(config, ckpt_path, args=args)  
        jmp_l_matbench_config_(config, targets, args.fold, base_path, args=args)
        init_model = MatbenchModel

    elif dataset_name == "qmof":
        config = QMOFConfig.draft()
        if args.model_name == "gemnet":
            jmp_l_ft_config_(config, ckpt_path, args=args)  
            jmp_l_qmof_config_(config, base_path, target=targets, args=args)
        elif args.model_name == "equiformer_v2":
            equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_qmof_config_(config, base_path, targets=targets, args=args)
        else:
            raise ValueError(f"Unsupported model for QMOF: {args.model_name}")
        init_model = QMOFModel

    elif dataset_name == "hmof":
        config = HMofConfig.draft()
        if args.model_name == "equiformer_v2":
            equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_hmof_config_(config, base_path, targets=targets, args=args)
            init_model = HMofModel

    elif dataset_name == "hmof-mini":
        config = HMofConfig.draft()
        if args.model_name == "equiformer_v2":
            equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_hmof_config_(config, base_path, targets=targets, args=args)
            init_model = HMofModel

    elif dataset_name == "obelix":
        config = ObelixConfig.draft()
        if args.model_name == "equiformer_v2":
            equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_obelix_config_(config, base_path, targets=targets, args=args)
            init_model = ObelixModel

    # elif dataset_name == "adsorption-db1":
    #     config = AdsorptionDbConfig.draft()
    #     jmp_l_ft_config_(config, ckpt_path, args=args)  
    #     jmp_l_adsorption_db_config_(config, targets, base_path)
    #     init_model = AdsorptionDbModel

    # elif dataset_name == "adsorption-db2":
    #     config = AdsorptionDbConfig.draft()
    #     jmp_l_ft_config_(config, ckpt_path, args=args)  
    #     jmp_l_adsorption_db_config_(config, targets, base_path)
    #     init_model = AdsorptionDbModel
        
    elif dataset_name in [ "adsorption_db1_merged", "adsorption_db2_merged", "adsorption_cof_db2"]:
        config = AdsorptionDbConfig.draft()
        config.meta["csv_path"] = None
        # (
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
        elif args.model_name == "equiformer_v2" and not args.llama and args.adabin:
            equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_adsorption_db_config_adabin_(config, targets, base_path, args)
        elif args.model_name == "equiformer_v2" and args.llama and args.adabin:
            autoreg_llama_equiformer_v2_ft_config_(config, ckpt_path, args=args)
            autoreg_llama_equiformer_v2_adsorption_db_config_adabin_(config, targets, base_path, args)
        elif args.model_name == "equiformer_v2" and not args.llama: 
            equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_adsorption_db_config_(config, targets, base_path, args)
        elif args.model_name == "equiformer_v2" and args.llama:
            autoreg_llama_equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_adsorption_db_config_(config, targets, base_path, args)
        
        init_model = AdsorptionDbModel
        
    elif not dataset_name and args.tasks:
        config = AdsorptionDbConfig.draft()
        
        if args.model_name == "gemnet":
            jmp_l_ft_config_(config, ckpt_path, args=args)
            jmp_l_adsorption_db_multi_task_config_(config, targets, args=args)
        if args.model_name == "equiformer_v2":
            equiformer_v2_ft_config_(config, ckpt_path, args=args)
            equiformer_v2_adsorption_db_multitask_config_(config, targets, args=args)
        
        init_model = AdsorptionDbModel

        # from jmp.models.EScAIP.config import GlobalConfigs
        # from pydantic import TypeAdapter
        # print("\n\n\n\n")
        # print("Pydantic type check:", TypeAdapter(GlobalConfigs).validate_python(config.backbone.global_cfg))
        # print("\n\n\n\n")
    elif dataset_name == "adsorption-cof-db1":
        assert False, "Not implemented yet"
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config.args = args
    

    try:
        config = config.finalize()
    except ValidationError as e:
        missing_fields = [error['loc'][0] for error in e.errors() if error['type'] == 'missing']
        print("Missing fields:", missing_fields)

    # We create checkpoints callback manually
    config.trainer.checkpoint_last_by_default = False
    config.trainer.on_exception_checkpoint = False
    
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
            new_weights[:old_weights.shape[0], :] = old_weights  # Copy old weights
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

def define_missing_keys_early_fusion(model, backbone_state):
    # PATCH: Handle new MOF projection layers not in checkpoint
    # Get the current state of the model (with initialized random weights for the new layers)
    current_model_state = model.backbone.state_dict()
    
    # Identify the keys for your new module
    # also add mof_rbf_encoder
    new_layer_prefixes = ["mof_embedding_proj", "mof_rbf_encoder"]
    
    # Loop through the current model keys
    for key in current_model_state:
        # If this key belongs to the new layer AND is missing from the loaded checkpoint
        if any(key.startswith(prefix) for prefix in new_layer_prefixes) and key not in backbone_state:
            print(f"Injecting missing key into state_dict: {key} (Initialized from scratch)")
            # Copy the random initialization from the current model into the dictionary we are loading
            backbone_state[key] = current_model_state[key]

    return backbone_state

def load_checkpoint(model, config, args):
    # Here, we load our fine-tuned model on the same task (including the head)

    
    # Here, we load Meta-AI original foundation model (without the heads)
    # else:

    if (ckpt_path := config.meta.get("ckpt_path")) is None:
        raise ValueError("No checkpoint path provided")
    print("Loading pretraining checkpoint =================", ckpt_path)

    # Load finetuned model
    if args.checkpoint_path:
        print("Loading custom checkpoint =================")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # 1. Load normal model weights
        state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict)

        # 2. Load EMA into the model
        if args.model_name == "equiformer_v2" and config.meta.get("ema_backbone", False):
            load_equiformer_ema_weights(ckpt, model)

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

        if "autoreg" in args.checkpoint_tag:
            print("Loading autoregressive checkpoint - removing heads and position embeddings")
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("backbone.coord_head.")}
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("backbone.atomic_num_head.")}
            # For fully autoregressive models, we do not load the position embeddings
            state_dict = {k: v for k, v in state_dict.items() if not "bos_embedding" in k}
            state_dict = {k: v for k, v in state_dict.items() if not "eos_embedding" in k}
            state_dict = {k: v for k, v in state_dict.items() if not "pos_embedding" in k}
        
        # this is heavily hacked here.
        elif "flowmatching" in args.checkpoint_tag:
            # remove the "equiformer." prefix and filter out flow matching specific keys
            new_state_dict = {}
            flow_matching_prefixes = ["backbone.time_embedding.", "backbone.velocity_coords_block."]
            
            for k, v in state_dict.items():
                # Skip flow matching specific parameters
                if any(k.startswith(prefix) for prefix in flow_matching_prefixes):
                    print(f"Skipping flow matching parameter: {k}")
                    continue
                    
                # remove "equiformer." prefix from backbone parameters
                if k.startswith("backbone.equiformer."):
                    new_key = k.replace("equiformer.", "", 1)
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
                    
            state_dict = new_state_dict

        if args.checkpoint_tag in ["jmp", "oc20_public", "oc20_2M_autoreg_onlyPos", "oc20_All_autoreg_onlyPos"]:
            new_embedding_shape = model.backbone.sphere_embedding.weight.shape
            state_dict = expand_embedding_state_dict(state_dict, new_embedding_shape)

        backbone_state = filter_state_dict(state_dict, "backbone.")

        if args.enable_feature_fusion and args.fusion_type == "early":
            backbone_state = define_missing_keys_early_fusion(model, backbone_state)

        model.backbone.load_state_dict(backbone_state)

        # Load EMA into the model
        if config.meta.get("ema_backbone", False):
            load_equiformer_ema_weights(ckpt, model)

def configure_wandb(config, args):
    """Configure WandB settings."""
    if args.enable_wandb:

        # Only rank 0 initializes WandB
        if not is_rank_zero():
            print("======== Disabling WandB logging for non-rank 0 process ========")
            config.trainer.logging.wandb.enabled = False
            config.trainer.auto_set_default_root_dir = False
            return
    
        # Datasets that should map to aramco_dac
        finetuning_datasets = {"rmd17", "md22", "qm9", "spice", "matbench", "qmof"}

        # Decide which project to use
        if args.dataset_name:
            project_name = (
                "aramco_dac" if any(ds in args.dataset_name for ds in finetuning_datasets)
                else "aramco_dac_finetune"
            )
        elif args.tasks:
            project_name = (
                "aramco_dac" if any(ds in " ".join(args.tasks) for ds in finetuning_datasets)
                else "aramco_dac_finetune"
            )
        else:
            raise ValueError("You must specify either --dataset_name or --tasks")

        # Define WandB parameters
        config.project = project_name
        config.entity = "theme4"
    else:
        # Disable wandb logging (for debugging)
        config.trainer.logging.wandb.enabled = False