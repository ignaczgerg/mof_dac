
# db1
# DATASET_PATH="/ibex/project/c2261/datasets/adsorption-db1/"
# RAW_PATH=$DATASET_PATH"raw/ADSORPTION-DB1-16-10-2024/"
# CSV_PATH=$RAW_PATH"5419_QMOF_DATABASE.csv"
# SPLIT_PATH=$DATASET_PATH"split_keys/"

#db2
# DATASET_PATH="/ibex/project/c2261/datasets/adsorption-db2/"
# RAW_PATH=$DATASET_PATH"raw/ADSORPTION-DB2-30-10-2024/"
# CSV_PATH=$RAW_PATH"7541_MOFs.csv"
# SPLIT_PATH=$DATASET_PATH"split_keys/"

#mof-db1
# DATASET_PATH="/ibex/project/c2261/datasets/adsorption-mof-db1/"
# RAW_PATH=$DATASET_PATH"raw/"
# CSV_PATH=$RAW_PATH"MOF-DB-1.0_DATABASE_28012025.csv"
# SPLIT_PATH=$DATASET_PATH"split_keys/"

# #mof-db2
# DATASET_PATH="/home/majedha/dev/datasets/adsorption-mof-db2/"
# RAW_PATH=$DATASET_PATH"raw/"
# CSV_PATH=$RAW_PATH"MOF_DB_2_0_18042025.csv"
# SPLIT_PATH=$DATASET_PATH"split_keys/"

# python -m jmp.datasets.scripts.adsorption_mof_db1_preprocess.adsorption_db1_splits --input_file $CSV_PATH --train_keys_output $SPLIT_PATH/"train_keys.pkl" --val_keys_output $SPLIT_PATH/"val_keys.pkl" --test_keys_output $SPLIT_PATH/"test_keys.pkl"

# # dump mof_cif_map for mof-db1 (not needed for mof-db2)
# python -m jmp.datasets.scripts.adsorption_mof_db1_preprocess.adsorption_db1_dataloader --dataset_path $RAW_PATH 


# generate lmdb
# python -m jmp.datasets.scripts.adsorption_mof_db1_preprocess.adsorption_db1_write_lmdbs --data-path $RAW_PATH --out-path $DATASET_PATH"lmdb/train" --split_keys $SPLIT_PATH/"train_keys.pkl" --num_workers 30

# python -m jmp.datasets.scripts.adsorption_mof_db1_preprocess.adsorption_db1_write_lmdbs --data-path $RAW_PATH --out-path $DATASET_PATH"lmdb/val" --split_keys $SPLIT_PATH/"val_keys.pkl" --num_workers 10

# python -m jmp.datasets.scripts.adsorption_mof_db1_preprocess.adsorption_db1_write_lmdbs --data-path $RAW_PATH --out-path $DATASET_PATH"lmdb/test" --split_keys $SPLIT_PATH/"test_keys.pkl" --num_workers 10


#mof-db2
DATASET_PATH="/ibex/project/c2261/datasets/adsorption-mof-ai-generated-2_0/"
RAW_PATH=$DATASET_PATH"cif/"

python -m jmp.datasets.scripts.adsorption_mof_db1_preprocess.adsorption_db1_write_lmdbs --cifs_path $RAW_PATH --out-path $DATASET_PATH"lmdb/test" --num_workers 10
