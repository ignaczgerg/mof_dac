cd ..
MAX_ATOMS_PER_BATCH=2000
python finetune.py \
    --dataset_name "hmof-mini" \
    --targets "qst_co2" \
    --normalization_type "standard" \
    --graph_scalar_reduction "mean" \
    --atom_bucket_batch_sampler \
    --num_workers 4 \
    --batch_size 4 \
    --lr 8e-5 \
    --epochs 15 \
    --model_name "equiformer_v2" \
    --small \
    --weight_decay 5e-3 \
    --dropout 0.0 \
    # --sratch \
    # --enable_wandb \
    # --checkpoint_tag "${CHECKPOINT_TAGS_ALL[$SLURM_ARRAY_TASK_ID]}" \
    # --checkpoint_tag "oc20_public" \
    # --scratch \
    # $POSITION_NORM