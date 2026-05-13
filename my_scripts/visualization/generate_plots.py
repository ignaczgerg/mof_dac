import os
import pandas as pd
import numpy as np
from plots_functions import (
    load_predictions,
    plot_target_histograms,
    plot_error_comparison,
    plot_training_size_impact,
    plot_metal_analysis,
    plot_scatter_gt_vs_pred,
    plot_PLD_LCD,
    plot_hierarchical_histograms_grid,
    plot_histograms_grid,
    plot_error_distribution

)

# ============================
# Create "plots" Folder if it Doesn't Exist
# ============================
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)  # Creates "plots/" if it does not exist

# ============================
# Load Dataset
# ============================
if os.path.exists("/ibex"):
    DATABASE_FILE = "/ibex/project/c2261/datasets/adsorption-mof-db1/raw/MOF-DB-1.0_DATABASE_28012025.csv"
elif os.path.exists("/home/majedha"):
    DATABASE_FILE = "/home/majedha/dev/datasets/adsorption-mof-db1/raw/MOF-DB-1.0_DATABASE_28012025.csv"
else:
    DATABASE_FILE = "/home/ghunaiym/MSI_resources/datasets/adsorption-mof-db1/raw/MOF-DB-1.0_DATABASE_28012025.csv"
dataset_df = pd.read_csv(DATABASE_FILE, encoding="ISO-8859-1")


# ============================
# Plot Selector (Enable/Disable)
# ============================
SELECT_PLOTS = {
    "targets_histograms": True, # USED
    "plot_error_comparison": True, # USED
    "plot_training_size_impact": True, # USED
    "metal_oms_analysis": True, # USED
    "pld_lcd_scatter": True, # Backup
    "gt_vs_pred_scatter": True, # USED
    "compare_models_histograms": True, # Not Used
    "metal_type_histograms": True # Not Used
}

# ============================
# Heat Adsorbtion Histograms
# ============================
if SELECT_PLOTS["targets_histograms"]:
    model_name = "gin"
    target = "qst_co2"
    train_df, val_df, test_df = load_predictions(dataset_df, model_name=model_name, target=target)

    plot_target_histograms(train_df)


# ============================
# Main Plot - Error Comparison
# ============================
if SELECT_PLOTS["plot_error_comparison"]:
    models = ["gin", "gcn", "schnet", "GemNetOC"]
    targets = ["qst_co2", "qst_h2o", "qst_n2"]

    plot_error_comparison(dataset_df, models, targets, split="test", error_type="MAE")
    plot_error_comparison(dataset_df, models, targets, split="test", error_type="MAPE")

# ============================
# Impact of training size
# ============================
if SELECT_PLOTS["plot_training_size_impact"]:
    models = ["gin", "gcn", "schnet", "GemNetOC"]
    targets = "qst_h2o"
    plot_training_size_impact(dataset_df, models=models, target=targets, split="test", error_type="MAE")
    plot_training_size_impact(dataset_df, models=models, target=targets, split="test", error_type="MAPE")

    targets = "qst_co2"
    plot_training_size_impact(dataset_df, models=models, target=targets, split="test", error_type="MAE")
    plot_training_size_impact(dataset_df, models=models, target=targets, split="test", error_type="MAPE")


# ============================
# Run Metal & OMS Analysis
# ============================
if SELECT_PLOTS["metal_oms_analysis"]:
    MODELS = ["GemNetOC"]
    TARGETS = ["qst_co2"]#, "co2_uptake"]
    COLUMNS = ["Unique Metal Entities"]#, "OMS"]
    MAX_VALUES = [(18, 18)]#[26.3, 1.07, 23, 1.4]
    MIN_VALUES = [(5, 10)]

    for model_name in MODELS:
        for target, max_val, min_val in zip(TARGETS, MAX_VALUES, MIN_VALUES):
            for column in COLUMNS:
                train_df, val_df, test_df = load_predictions(dataset_df, model_name=model_name, target=target)

                for split, df in zip(["train", "test"], [train_df, test_df]):
                    column_name = column
                    if column == "Unique Metal Entities":
                        column_name = "Metal"
                    max_value = max_val[0] if split=="train" else max_val[1]
                    min_value = min_val[0] if split=="train" else min_val[1]
                    save_path = f"plots/{column_name}_{model_name}_{target}_{split}.png"
                    plot_metal_analysis(df, column=column, title=f"{split.capitalize()} {column_name} Analysis ({target})",
                                        max_value=max_value, min_value=min_value, save_path=save_path)

# ============================
# PLD/LCD Scatter Plot
# ============================
if SELECT_PLOTS["pld_lcd_scatter"]:
    model_name = "GemNetOC"
    target = "qst_co2"
    train_df, val_df, test_df = load_predictions(dataset_df, model_name=model_name, target=target)

    # metal = "Cu"  # Replace with any desired metal
    # train_df = train_df[train_df["Unique Metal Entities"] == metal]
    # test_df = test_df[test_df["Unique Metal Entities"] == metal]

    # Define x/y limits based on data range
    all_y_values = np.concatenate([
        train_df["ground_truth"], test_df["ground_truth"],
        train_df["predicted_value"], test_df["predicted_value"]
    ])
    y_l, y_h = all_y_values.min() - 5, 10
    xlim, ylim = [0, 50], [y_l, y_h]

    # Generate the plot
    plot_PLD_LCD(train_df, test_df, model_name, target, xlim, ylim,
                save_path=f"plots/PLD_LCD_{model_name}_{target}.png")

# ============================
# Scatter Plot: GT vs Prediction
# ============================
if SELECT_PLOTS["gt_vs_pred_scatter"]:
    for model_name in ["gin", "gcn", "schnet", "GemNetOC"]:
    # model_name = "gcn"
        target = "qst_co2"
        train_df, val_df, test_df = load_predictions(dataset_df, model_name=model_name, target=target)

        all_y_values = np.concatenate([
            train_df["ground_truth"], test_df["ground_truth"],
            train_df["predicted_value"], test_df["predicted_value"]
        ])
        y_l, y_h = all_y_values.min() - 5, all_y_values.max() + 5
        xlim, ylim = [-60, 0], [-60, 0]  # Modify if needed

        plot_scatter_gt_vs_pred(train_df, test_df, model_name, target, xlim, ylim,
                                save_path=f"plots/gt_vs_pred_{model_name}_{target}.png")


# ============================
# Histograms Plot: Compare Models
# ============================
if SELECT_PLOTS["compare_models_histograms"]:
    MODELS_TO_COMPARE = ["gin", "gcn", "schnet", "GemNetOC"]
    target = "qst_co2"

    def rename(model):
        return {"gin": "GIN", "gcn": "GCN", "schnet": "SchNet", "GemNetOC": "GemNet-OC"}[model]

    for split in ["train", "test"]:
        models_dfs = {
            rename(model): load_predictions(dataset_df, model_name=model, target=target)[0 if split == "train" else 2]
            for model in MODELS_TO_COMPARE
        }

        plot_hierarchical_histograms_grid(
            models_dfs=models_dfs,
            y_column_gt="ground_truth",
            y_column_pred="predicted_value",
            title=f"Qst $CO_2$ Prediction Histograms - {split.capitalize()}",
            save_path=f"plots/histograms_grid_qst_co2_{split}.png",
            starting_bin=-70, ending_bin=0
        )


# ============================
# Histograms: GemNet-OC for Different Metals
# ============================
if SELECT_PLOTS["metal_type_histograms"]:
    SELECTED_METALS = ["Zn", "Fe", "Cu", "Ag", "Cd"]
    target = "qst_co2"
    model_name = "GemNetOC"

    for split in ["test"]:#["train", "test"]:
        models_dfs = {}
        for metal in SELECTED_METALS:
            train_df, val_df, test_df = load_predictions(dataset_df, model_name=model_name, target=target)
            train_df = train_df.query("`Unique Metal Entities` == @metal")
            val_df = val_df.query("`Unique Metal Entities` == @metal")
            xlim = None
            ylim = None
            models_dfs[f"GemNetOC - {metal}"] = val_df

        plot_error_distribution(#plot_histograms_grid(
            models_dfs=models_dfs,
            y_column_gt="ground_truth",
            y_column_pred="predicted_value",
            error_type="MAE",
            title=f"Qst $CO_2$ Prediction Histograms - {split.capitalize()}",
            save_path=f"plots/histograms_grid_qst_co2_{split}.png",
            # starting_bin=-70, ending_bin=0
        )

           
        # models_dfs = {
        #     f"GemNetOC - {metal}": load_predictions(dataset_df, model_name=model_name, target=target)[0 if split == "train" else 1].query("`Unique Metal Entities` == @metal")
            
        # }
        # models_dfs = {
        #     f"GemNetOC - {metal}": load_predictions(dataset_df, model_name=model_name, target=target)[0 if split == "train" else 1].query("`Unique Metal Entities` == @metal")
        #     for metal in SELECTED_METALS
        # }

        # plot_hierarchical_histograms_grid(
        #     models_dfs=models_dfs,
        #     y_column_gt="ground_truth",
        #     y_column_pred="predicted_value",
        #     title=f"Qst $CO_2$ Prediction Histograms - {split.capitalize()} (by Metal Type)",
        #     save_path=f"plots/histograms_grid_qst_co2_metals_{split}.png",
        #     starting_bin=-60, ending_bin=0
        # )

        # plot_histograms_grid(
        #     models_dfs=models_dfs,
        #     y_column_gt="ground_truth",
        #     y_column_pred="predicted_value",
        #     title=f"Qst $CO_2$ Prediction Histograms - {split.capitalize()}",
        #     save_path=f"plots/histograms_grid_qst_co2_{split}.png",
        #     # starting_bin=-70, ending_bin=0
        # )

        
