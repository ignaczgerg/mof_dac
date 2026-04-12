import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from bokeh.io import export_png, curdoc
from periodic_table import plotter_with_counts
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score

plt.style.use('science')

# ============================
# Metal Group Classification
# ============================
METAL_GROUPS = {
    "Alkali Metals": {"Li", "Na", "K", "Rb", "Cs", "Fr"},
    "Alkaline Earth Metals": {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"},
    "Transition Metals": {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"},
    "Post-Transition Metals": {"Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi"},
    "Metalloids": {"B", "Si", "Ge", "As", "Sb", "Te", "Po"},
    "Noble Gases": {"He", "Ne", "Ar", "Kr", "Xe", "Rn"},
    "Lanthanides": {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"},
    "Actinides": {"Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}
}

METAL_TO_GROUP = {metal: group for group, metals in METAL_GROUPS.items() for metal in metals}

def get_metal_group(metal_str: str) -> str:
    """Classify a metal entry into its respective group."""
    metals = {m.strip() for m in metal_str.split(",")}
    groups = {METAL_TO_GROUP.get(m, "Unknown") for m in metals}
    return next(iter(groups)) if len(groups) == 1 else "Mixed"

# ============================
# Data Loading
# ============================
def load_predictions(dataset_df: pd.DataFrame, model_name: str, target: str, results_path: str = "results/") -> tuple:
    """Load prediction CSV files and filter the main dataset accordingly."""
    train_prediction = pd.read_csv(f"{results_path}{model_name}_{target}_train.csv")
    val_prediction = pd.read_csv(f"{results_path}{model_name}_{target}_val.csv")
    
    try:
        test_prediction = pd.read_csv(f"{results_path}{model_name}_{target}_test.csv")
    except FileNotFoundError:
        test_prediction = None
        raise FileNotFoundError("Test prediction file not found.")


    # Merge predictions with the main dataset
    train_df = dataset_df[dataset_df["MOF ID"].isin(train_prediction["mof_id"])].merge(
        train_prediction, left_on="MOF ID", right_on="mof_id", how="inner"
    )
    val_df = dataset_df[dataset_df["MOF ID"].isin(val_prediction["mof_id"])].merge(
        val_prediction, left_on="MOF ID", right_on="mof_id", how="inner"
    )
    if test_prediction is None:
        test_df = None
    else:
        test_df = dataset_df[dataset_df["MOF ID"].isin(test_prediction["mof_id"])].merge(
            test_prediction, left_on="MOF ID", right_on="mof_id", how="inner"
        )

    return train_df, val_df, test_df


# ============================
# Target Histograms
# ============================
def plot_target_histograms(train_df):
    plt.style.use("science")  # Apply science plot style
    targets = ["Qst_N2(kJ/mol)", "Qst_H2O(kJ/mol)", "Qst_CO2(kJ/mol)"]

    for target in targets:
        plt.figure(figsize=(6, 6))
        
        # Histogram 
        filtered_data = train_df[target][(train_df[target] > train_df[target].quantile(0.0005)) & 
                                 (train_df[target] < train_df[target].quantile(0.9995))]

        sns.histplot(filtered_data, bins=30, color='#91bfdb', edgecolor='black', alpha=1)
        
        ax = plt.gca()

        plt.xticks(rotation=45)
        
        # Remove y-axis completely
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_visible(False)

        # Remove top x-axis and right spine
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Remove top xticks
        # plt.tick_params(axis='x', top=False)
        ax.tick_params(axis='x', which='both', top=False, labeltop=False)

        # Increase font for x ticks using ax

        plt.xticks(fontsize=30)

        # Formatting labels and title
        clean_target = target.replace("(kJ/mol)", "").strip()
        # plt.xlabel(rf"{clean_target} $\frac{{\mathrm{kJ}}}{{\mathrm{mol}}}$", fontsize=26)
        plt.xlabel(rf"{clean_target} " + r"($\frac{\mathrm{kJ}}{\mathrm{mol}}$)", fontsize=30)

        # Save figure
        filename = f"plots/histogram_{clean_target}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


# ============================
# Main Results (MAE and MAPE)
# ============================

def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error (MAE)."""
    return np.mean(np.abs(y_true - y_pred))

def compute_mape(y_true, y_pred):
    """Compute Mean Absolute Percentage Error (MAPE)."""
    mask = y_true != 0  # Avoid division by zero
    return 100*np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def plot_error_comparison(dataset_df, models, targets, split="test", error_type="MAE"):
    """
    Generates a bar plot for Mean Absolute Error (MAE) or Mean Absolute Percentage Error (MAPE).

    Args:
        dataset_df (pd.DataFrame): The full dataset containing predictions.
        models (list): List of model names to evaluate (e.g., ["gin", "gcn", "schnet", "gemnet-oc"]).
        targets (list): List of target properties to analyze (e.g., ["qst_co2", "qst_h2o", "qst_n2"]).
        split (str): The dataset split to compute error on. Options: ["train", "val", "test"].
        error_type (str): Type of error to compute. Options: ["MAE", "MAPE"].
    """
    error_data = []

    # Load predictions and compute MAE for each model & target
    for target in targets:
        for model_name in models:
            train_df, val_df, test_df = load_predictions(dataset_df, model_name=model_name, target=target)

            # Select the required split
            if split == "train":
                df = train_df
            elif split == "val":
                df = val_df
            elif split == "test":
                df = test_df
                if test_df is None:
                    raise ValueError("Test results are not found")
            else:
                raise ValueError(f"Invalid split: {split}. Choose from ['train', 'val', 'test'].")

            # Compute MAE if data is available
            if df is not None and not df.empty:
                y_true = df["ground_truth"]
                y_pred = df["predicted_value"]
                error = compute_mae(y_true, y_pred) if error_type == "MAE" else compute_mape(y_true, y_pred)
            else:
                error = np.nan  # Assign NaN if no data is available

            error_data.append({"Molecule": target, "Model": model_name, "Error": error})

    # Convert to DataFrame
    df = pd.DataFrame(error_data)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Molecule", y="Error", hue="Model", palette=["#F7E124", "#5AC360", "#1E8D8A", "#3B4F88"])

    # Add labels to bars
    for index, row in df.iterrows():
        y_offset = 0.2 if error_type == "MAE" else 0.5
        plt.text(x=targets.index(row["Molecule"]) + (-0.36 + 0.2 * models.index(row["Model"])),
                 y=row["Error"] + y_offset,
                 s=f"{row['Error']:.1f}",
                 fontsize=16, rotation=0)

    # Titles and labels
    plt.xlabel("")
    ylabel_text = "MAE (kJ/mol)" if error_type == "MAE" else "MAPE (\%)"
    plt.ylabel(ylabel_text, fontsize=20)
    

    plt.ylim(0, df["Error"].max() + 0.1 * df["Error"].max())

    # plt.minorticks_off()
    plt.tick_params(axis='x', which='both', top=False)
    plt.tick_params(axis='y', which='both', right=False)
    plt.yticks(fontsize=20)

    # Rename legend labels
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    label_mapping = {"gin": "GIN", "gcn": "GCN", "schnet": "SchNet"}
    new_labels = [label_mapping.get(label, label) for label in labels]
    ax.legend(handles, new_labels, title="", fontsize=20, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)


    # Rename x tick labels
    xticks = ax.get_xticks()
    xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    new_xticklabels = [name.split("_")[0].title() + " " + name.split("_")[1].upper() for name in xticklabels]
    ax.set_xticks(xticks)
    ax.set_xticklabels(new_xticklabels, fontsize=20)

    # Save
    filename = f"plots/{error_type.lower()}_comparison_{split}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")

# ============================
# Impact of training set size
# ============================

def plot_training_size_impact(dataset_df, models, target, split="test", error_type="MAE", results_path="results/"):
    """
    Analyzes the impact of different training sizes (5k, 10k, 15k) on model performance for a single target.

    Args:
        dataset_df (pd.DataFrame): The full dataset containing MOF ID mappings.
        models (list): List of model names (e.g., ["gin", "gcn", "schnet", "GemNetOC"]).
        target (str): The target property to analyze (e.g., "qst_co2").
        split (str): The dataset split to compute error on ["train", "val", "test"].
        error_type (str): Type of error to compute ["MAE", "MAPE"].
        results_path (str): Path to the results folder.
    """
    training_sizes = ["5000", "10000", "15000"]
    error_data = []

    for train_size in training_sizes:
        size_folder = train_size if train_size != "15000" else ""  # Use base folder for full dataset
        
        for model_name in models:
            # Determine file path
            file_suffix = f"_{train_size}.csv" if size_folder else ".csv"
            file_path = os.path.join(results_path, size_folder, f"{model_name}_{target}_{split}{file_suffix}")

            # Try loading the file
            if not os.path.exists(file_path):
                print(f"Skipping: {file_path} not found.")
                continue

            df = pd.read_csv(file_path)

            # Merge with dataset_df to get MOF ID mappings
            df = dataset_df[dataset_df["MOF ID"].isin(df["mof_id"])].merge(df, left_on="MOF ID", right_on="mof_id", how="inner")

            # Compute MAE or MAPE
            if not df.empty:
                y_true = df["ground_truth"]
                y_pred = df["predicted_value"]
                error = compute_mae(y_true, y_pred) if error_type == "MAE" else compute_mape(y_true, y_pred)
            else:
                error = np.nan  # Assign NaN if no data is available

            error_data.append({"Training Size": train_size, "Model": model_name, "Error": error})

    # Convert to DataFrame
    df_plot = pd.DataFrame(error_data)

    if error_type == "MAPE":
        print(df_plot)

    # Rename models for consistency
    model_mapping = {"gin": "GIN", "gcn": "GCN", "schnet": "SchNet", "GemNetOC": "GemNetOC"}
    df_plot["Model"] = df_plot["Model"].replace(model_mapping)
    models = [model_mapping.get(m) for m in models]


    # Sort training size for proper line connections
    df_plot["Training Size"] = df_plot["Training Size"].astype(int)
    df_plot = df_plot.sort_values("Training Size")
    
    # Define custom colors for each model
    model_colors = {
        "GIN": "#fc8d59",       
        "GCN": "#5AC360",       
        "SchNet": "#1E8D8A",    
        "GemNetOC": "#3B4F88"   
    }


    # Plot
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_plot, x="Training Size", y="Error", hue="Model", style="Model", markers=True, dashes=False, linewidth=2.5, markersize=16, palette=model_colors)

    # Titles and labels
    if target == "qst_h2o":
        name = "H2O"
    elif target == "qst_co2":
        name = "CO2"
    plt.xlabel(f"Training Size (Qst {name})", fontsize=30)
    ylabel_text = "MAE (kJ/mol)" if error_type == "MAE" else "MAPE (\%)"
    plt.ylabel(ylabel_text, fontsize=30)

    if error_type == "MAE" and target == "qst_co2":
        plt.ylim(3, 7)

    # Customize ticks
    plt.xticks([5000, 10000, 15000], labels=["5K", "10K", "15k"], fontsize=26)
    plt.yticks(fontsize=26)
    plt.minorticks_off()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Improve legend placement
    plt.legend(loc='upper center', fontsize=26, bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False)


    # Save
    # plt.tight_layout()
    filename = f"plots/{error_type.lower()}_training_size_{target}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")


# ============================
# Metal Analysis Plot
# ============================
def plot_metal_analysis(df: pd.DataFrame, column: str, title: str, max_value: int, min_value:int, save_path: str) -> None:
    """Generate and save a metal analysis plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not save_path.endswith(".png"):
        raise ValueError("save_path must be a PNG file (e.g., 'results/metal_analysis.png').")

    # df["absolute_error"] = (df["ground_truth"] - df["predicted_value"]).abs()
    df["absolute_error"] = 100 * ((df["ground_truth"] - df["predicted_value"]).abs() / df["ground_truth"].abs())

    df_filtered = df[~df[column].str.contains(",", na=False)]

    if column == "OMS":
        df_filtered = df_filtered[df_filtered[column] != "No"]

    mae_per_metal = df_filtered.groupby(column)["absolute_error"].mean()
    occurrence_per_metal = df_filtered[column].value_counts()

    combined_data = pd.DataFrame({"MAE": mae_per_metal, "Count": occurrence_per_metal}).reset_index()
    combined_data.columns = ["Element", "MAE", "Count"]

    curdoc().clear()
    fig = plotter_with_counts(combined_data, max_value, min_value)
    fig.title.text = title

    export_png(fig, filename=save_path)
    print(f"Figure saved at: {save_path}")

    # Remove the auto-generated HTML file
    html_path = f"generate_plots.html"
    if os.path.exists(html_path):
        os.remove(html_path)
        print(f"Temporary HTML file removed: {html_path}")

# ============================
# Scatter Plot: Ground Truth vs Prediction
# ============================
def plot_scatter_gt_vs_pred(train_df: pd.DataFrame, test_df: pd.DataFrame, model_name: str, target: str, xlim: tuple, ylim: tuple, save_path: str) -> None:
    """
    Generate a 2-row scatter plot grid for ground truth vs predicted values.
    - Includes a reference line (perfect predictions).
    - Uses colormap to highlight errors.
    - Displays R² score.

    Args:
        train_df (pd.DataFrame): Training set dataframe.
        test_df (pd.DataFrame): Test set dataframe.
        model_name (str): Model name for labeling.
        target (str): Target property.
        xlim (tuple): X-axis limits.
        ylim (tuple): Y-axis limits.
        save_path (str): Path to save the figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Standardized model name mapping
    label_mapping = {"gin": "GIN", "gcn": "GCN", "schnet": "SchNet", "gemnet-oc": "GemNet-OC"}
    model_name = label_mapping.get(model_name, model_name)  # Use mapped name if available

    fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharex=False, sharey=False)  # Larger figure size

    for ax, (df, split) in zip(axes.flat, [(train_df, "Train"), (test_df, "Test")]):
        if df.empty:
            ax.set_title(f"No Data for {split}", fontsize=26, fontweight="bold")
            ax.axis("off")
            continue

        # Compute absolute error
        df["Error"] = np.abs(df["ground_truth"] - df["predicted_value"])

        # Compute R² score
        r2 = r2_score(df["ground_truth"], df["predicted_value"])

        # Normalize error for colormap (avoid extreme values skewing visualization)
        vmin, vmax = 0, 8  # Clip max error for better visualization
        scatter = ax.scatter(df["ground_truth"], df["predicted_value"], 
                             c=df["Error"], cmap="coolwarm", alpha=0.8, edgecolors="black", s=100,
                             vmin=vmin, vmax=vmax)  # Normalized color range

        # Add perfect prediction reference line
        ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], color="black", linestyle="dashed", linewidth=2, label="Perfect Prediction")

        # Formatting and Titles
        ax.set_xlabel("Ground Truth", fontsize=32, fontweight="bold")
        ax.set_ylabel("Predicted Value", fontsize=32, fontweight="bold")
        # ax.set_title(f"{split} Set", fontsize=32, fontweight="bold", pad=15)
        ax.grid(True, linestyle="--", alpha=0.5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Increase tick size
        ax.tick_params(axis='both', which='major', labelsize=30)

        # Add R² text annotation
        ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, fontsize=30,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

        # Add colorbar with larger font size
        cbar = plt.colorbar(scatter, ax=ax, orientation="vertical")
        cbar.set_label("Absolute Error", fontsize=26, fontweight="bold")
        cbar.ax.tick_params(labelsize=22)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved at: {save_path}")



# ============================
# Scatter Plot: PLD/LCD vs Predictions
# ============================
def plot_PLD_LCD(train_df: pd.DataFrame, test_df: pd.DataFrame, model_name: str, target: str, xlim: tuple, ylim: tuple, save_path: str) -> None:
    """Generate scatter plots comparing PLD/LCD values with ground truth and predicted values."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # Increased figure size for better separation
    plot_params = [(train_df, "PLD ", "Train"), (train_df, "LCD ", "Train"),
                   (test_df, "PLD ", "Test"), (test_df, "LCD ", "Test")]

    for idx, (ax, (df, col, split)) in enumerate(zip(axes.flat, plot_params)):
        if df.empty:
            ax.set_title(f"No data for {split}", fontsize=24, fontweight="bold")
            ax.axis("off")
            continue

        scatter_gt = ax.scatter(df[col], df["ground_truth"], color="#1E8D8A", alpha=0.6, edgecolors="black", s=80, label="Ground Truth")
        scatter_pred = ax.scatter(df[col], df["predicted_value"], color="#fc8d59", alpha=0.6, edgecolors="black", s=80, label="Prediction")
        
        ax.set_xlabel(f"{col} (Å)", fontsize=28, fontweight="bold")
        ax.set_ylabel("Values", fontsize=28, fontweight="bold")
        ax.set_title(f"{split}", fontsize=30, fontweight="bold", pad=20)

        # Increase y-axis tick count
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))  # Increased number of major ticks

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major', labelsize=24)

    # Add a single legend outside the subplots
    handles = [scatter_gt, scatter_pred]
    labels = ["Ground Truth", "Prediction"]
    fig.legend(handles, labels, fontsize=26, loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=2, frameon=False)

    if target == "qst_co2":
        target = "Qst CO2"
    plt.suptitle(f"PLD and LCD Analysis for {model_name} - {target}", fontsize=36, fontweight="bold", y=1.05)

    # Increase space between plots
    plt.subplots_adjust(hspace=0.6, wspace=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved at: {save_path}")


# ============================
# Hierarchical Histograms Grid
# ============================
def plot_hierarchical_histograms_grid(
    models_dfs: dict, y_column_gt: str, y_column_pred: str, 
    title: str, save_path: str, starting_bin: int = -80, ending_bin: int = 10
) -> None:
    """Generates a 4-row grid of hierarchical histograms comparing ground truth vs predicted values for multiple models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    bins = np.arange(starting_bin, ending_bin, 10)
    bin_labels = [f"{bins[i]} to {bins[i+1]}" for i in range(len(bins) - 1)]

    fig, axes = plt.subplots(len(models_dfs), len(bin_labels), figsize=(3 * len(bin_labels), 4 * len(models_dfs)), sharey=False)
    fig.suptitle(title, fontsize=50, fontweight='bold')

    for row, (model, df) in enumerate(models_dfs.items()):
        model_ax = fig.add_subplot(len(models_dfs), 1, row + 1, frame_on=False)
        model_ax.set_xticks([])
        model_ax.set_yticks([])
        model_ax.set_title(model, fontsize=40, fontweight='bold', pad=30)

        max_y = 0
        for col, label in enumerate(bin_labels):
            bin_min, bin_max = bins[col], bins[col + 1]
            bin_data = df[(df[y_column_gt] >= bin_min) & (df[y_column_gt] < bin_max)]

            ax = axes[row, col] if len(models_dfs) > 1 else axes[col]

            if not bin_data.empty:
                gt_counts, gt_edges = np.histogram(bin_data[y_column_gt], bins=30)
                pred_counts, pred_edges = np.histogram(bin_data[y_column_pred], bins=30)

                gt_percent = gt_counts / gt_counts.sum() * 100 if gt_counts.sum() > 0 else gt_counts
                pred_percent = pred_counts / pred_counts.sum() * 100 if pred_counts.sum() > 0 else pred_counts

                ax.bar(gt_edges[:-1], gt_percent, alpha=0.6, color='blue', label='Ground Truth', align='edge')
                ax.bar(pred_edges[:-1], pred_percent, alpha=0.6, color='red', label='Prediction', align='edge')

                max_y = max(max_y, ax.get_ylim()[1])

            ax.set_title(label, fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=18, left=(col == 0), labelleft=(col == 0))
            ax.grid(True, linestyle="--", alpha=0.5)
            if col == 0:
                ax.set_ylabel("Percentage (%)", fontsize=24)

            ax.minorticks_off()
            ax.tick_params(axis='x', which='both', top=False)
            ax.tick_params(axis='y', which='both', right=False)

        for col in range(len(bin_labels)):
            max_y = min(max_y, 30)
            axes[row, col].set_ylim(0, max_y)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=24, ncol=2, bbox_to_anchor=(0.5, 0.93))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=1.0, wspace=0.08)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved at: {save_path}")
    # plt.show()


# ============================
# Histograms Grid
# ============================
def plot_histograms_grid(
    models_dfs: dict, y_column_gt: str, y_column_pred: str, 
    title: str, save_path: str, starting_bin: int = -80, ending_bin: int = 10
) -> None:
    """Generates a 4-row grid of hierarchical histograms comparing ground truth vs predicted values for multiple models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(len(models_dfs), 1, figsize=(10, 4 * len(models_dfs)), sharey=False, sharex=False)
    fig.suptitle(title, fontsize=50, fontweight='bold')

    for row, (model, df) in enumerate(models_dfs.items()):
        # model_ax = fig.add_subplot(len(models_dfs), 1, row + 1, frame_on=False)
        # model_ax.set_xticks([])
        # model_ax.set_yticks([])
        # model_ax.set_title(model, fontsize=40, fontweight='bold', pad=30)


        # max_y = 0
        ax = axes[row]
        # gt_counts, gt_edges = np.histogram(df[y_column_gt], bins=30)
        # pred_counts, pred_edges = np.histogram(df[y_column_pred], bins=30)

        ax.hist(df[y_column_gt], bins=50, alpha=0.5, label="Ground Truth")
        ax.hist(df[y_column_pred], bins=50, alpha=0.5, label="Predicted Value")

        # sns.histplot(df[y_column_pred], bins=30, color='#91bfdb', edgecolor='black', alpha=1)
        # ax.hist(gt_edges[:-1], gt_percent, alpha=0.6, color='blue', label='Ground Truth', align='edge')

        # gt_percent = gt_counts / gt_counts.sum() * 100 if gt_counts.sum() > 0 else gt_counts
        # pred_percent = pred_counts / pred_counts.sum() * 100 if pred_counts.sum() > 0 else pred_counts

        # ax.bar(gt_edges[:-1], gt_percent, alpha=0.6, color='blue', label='Ground Truth', align='edge')
        # ax.bar(pred_edges[:-1], pred_percent, alpha=0.6, color='red', label='Prediction', align='edge')
        # max_y = max(max_y, ax.get_ylim()[1])

        # ax.set_title(label, fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylabel("Percentage (\%)", fontsize=24)

        ax.minorticks_off()
        ax.tick_params(axis='x', which='both', top=False)
        ax.tick_params(axis='y', which='both', right=False)

        # ax.set_ylim(0, min(max_y, 30))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=24, ncol=2, bbox_to_anchor=(0.5, 0.93))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=1.0, wspace=0.08)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved at: {save_path}")

def plot_error_distribution(
    models_dfs: dict, y_column_gt: str, y_column_pred: str, 
    error_type: str, title: str, save_path: str
) -> None:
    """
    Generates a grid of histograms showing error distributions for different models.

    Args:
        models_dfs (dict): Dictionary where keys are model names and values are DataFrames.
        y_column_gt (str): Column name for ground truth values.
        y_column_pred (str): Column name for predicted values.
        error_type (str): "MAE" or "MAPE" to determine which error metric to compute.
        title (str): Title of the plot.
        save_path (str): Path to save the figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_models = len(models_dfs)
    fig, axes = plt.subplots(num_models, 1, figsize=(12, 4 * num_models), sharey=False)
    fig.suptitle(title, fontsize=30, fontweight='bold')

    if num_models == 1:
        axes = [axes]  # Ensure axes is iterable for a single model

    for ax, (model, df) in zip(axes, models_dfs.items()):
        # Compute error
        df["Error"] = df[y_column_gt] - df[y_column_pred]
        y_label = "Prediction Error (y_true - y_pred)"

        # Histogram for error distribution
        sns.histplot(df["Error"], bins=40, kde=True, color="#91bfdb", edgecolor="black", alpha=0.7, ax=ax)
        
        # Add KDE line to smooth the distribution and emphasize patterns
        sns.kdeplot(df["Error"], color="red", linewidth=2, ax=ax)

        # Annotate mean and standard deviation on the plot for context
        mean_error = df["Error"].mean()
        std_error = df["Error"].std()
        ax.axvline(mean_error, color='blue', linestyle='dashed', linewidth=2)
        ax.text(mean_error + 0.1, ax.get_ylim()[1] * 0.9, f"Mean: {mean_error:.2f}", color='blue', fontsize=12)
        ax.text(mean_error + 0.1, ax.get_ylim()[1] * 0.8, f"Std: {std_error:.2f}", color='blue', fontsize=12)

        # Set fixed x-limits
        ax.set_xlim(-10, 10)

        # Titles and formatting
        ax.set_title(f"{model}", fontsize=26, fontweight='bold', pad=20)
        ax.set_xlabel(y_label, fontsize=18)
        ax.set_ylabel("Count", fontsize=18)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=16)

    # Adjust layout to prevent squeezing and enhance spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.6)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved at: {save_path}")
