import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import scienceplots


# ------------------ CONFIG: CHANGE THESE PATHS ONLY ------------------
# Prediction files (AI-MOFs)
BASE_PRED_DIR = "/ibex/project/c2261/dac_iclr/finetune/lightning_logs/ywrlm13n/aramco_dac_finetune/ywrlm13n/checkpoints/predictions_external/external_eval/test_ai-mof-batch3_new"

# Prediction files (AI-MOFs)
normalization_task = "db_2" # choose "db_1" or "db_2"
batch_ID = "batch3"
PRED_PATHS = {
    name: f"{BASE_PRED_DIR}/{name}_{normalization_task}.npz"
    for name in ["co2_uptake", "kh_co2", "qst_co2"]
}

# Ground truth CSV
if batch_ID == "batch1":
    GT_CSV = "../evaluate_results/ai-generated_B1_gt.csv"
elif batch_ID == "batch3":
    GT_CSV = "../evaluate_results/ai-generated_B3_gt.csv"


# ------------------ STYLE ------------------
plt.style.use(['science', 'no-latex'])
colors = ["#96A2C3", "#DB96C0", "#E89676", "#71B6A1", "#ff9287", "#B1AF55",
          "#ff8234ff", "#ff9287", "#5954d6", "#00c6f8", "#878500", "#00a76c", "#bdbdbd"]



# ------------------ FUNCTIONS ------------------
def smape(x, y, eps=1e-8):
    x, y = np.array(x), np.array(y)
    denominator = (np.abs(x) + np.abs(y)) / 2.0
    return np.mean(np.abs(y - x) / (denominator + eps)) * 100


def plot_equity_with_band(ax, x, y, title, log=False, key=None):

    REGION_OF_INTEREST = {
        'qst_co2': [-30, -60],        
        'kh_co2':  [-3.5, 0],        
        'co2_uptake': [-2, 1],        
        'qst_h2o': [-40, -80],        
        'kh_h2o':  [-2, 1],           
        'selectivity_co2_h2o': [0, 3],
        }
    ax.scatter(x, y, s=0.5)
    r2 = r2_score(x, y)

    minv, maxv = x.min(), x.max()
    ax.plot([minv, maxv], [minv, maxv], 'k--', lw=1, label="Equity line")

    if log:
        upper = np.array([minv, maxv]) + 0.5
        lower = np.array([minv, maxv]) - 0.5
        _y = np.exp(y)
        _x = np.exp(x)
        smape_error = smape(_x, _y)

    else:
        upper = 1.1 * np.array([minv, maxv])
        lower = 0.9 * np.array([minv, maxv])
        smape_error = smape(x, y)        

    ax.plot([minv, maxv], upper, 'k:', lw=0.8, label="Error bounds")
    ax.plot([minv, maxv], lower, 'k:', lw=0.8)

    slope = (upper[1] - upper[0]) / (maxv - minv)
    upper_line = upper[0] + slope * (x - minv)
    lower_line = lower[0] + slope * (x - minv)

    outside = np.sum((y > upper_line) | (y < lower_line))

    ax.set_xlabel("Ground Truth" + (" (log10)" if log else ""))
    ax.set_ylabel("Prediction" + (" (log10)" if log else ""))
    ax.set_title(title, fontsize=8)
    if log:
        ax.text(0.05, 0.95, f"$R^2$ (log): {r2:.2f}", transform=ax.transAxes,va="top", ha="left", fontsize=8)
    else:
        ax.text(0.05, 0.95, f"$R^2$ (norm): {r2:.2f}", transform=ax.transAxes,va="top", ha="left", fontsize=8 )
    if key is not None and key in REGION_OF_INTEREST:
        rx0, rx1 = REGION_OF_INTEREST[key]
        ry0, ry1 = REGION_OF_INTEREST[key]
        x0, x1 = (rx0, rx1) if rx0 <= rx1 else (rx1, rx0)
        y0, y1 = (ry0, ry1) if ry0 <= ry1 else (ry1, ry0)

        ax.fill_between([rx0, rx1], ry0, ry1, color="lightblue", alpha=0.15)
        ax.add_patch(
            plt.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                          fill=False, edgecolor="lightblue", lw=1.0, linestyle="--")
        )

        roi_mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        in_roi = int(roi_mask.sum())
        r2_roi = np.nan
        if in_roi >= 2:
            r2_roi = r2_score(x[roi_mask], y[roi_mask])

        ax.text(0.05, 0.85,
                f"$R^2$ (ROI): {r2_roi:.2f}" if in_roi >= 2 else "$R^2$ (ROI): -",
                transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.text(0.05, 0.73,
                f"In ROI: {in_roi/len(y):.0%}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8)

    if key is not None and key in REGION_OF_INTEREST:
        roi_xmin, roi_xmax = REGION_OF_INTEREST[key]
        roi_ymin, roi_ymax = REGION_OF_INTEREST[key]
        ax.fill_between(
            [roi_xmin, roi_xmax],
            roi_ymin, roi_ymax,
            color="lightblue", alpha=0.2,
        )
        ax.add_patch(
            plt.Rectangle(
                (roi_xmin, roi_ymin),
                roi_xmax - roi_xmin,
                roi_ymax - roi_ymin,
                fill=False, edgecolor="lightblue", lw=1.0, linestyle="--"
            )
        )
    if log:
        ax.text(
            x=roi_xmax,
            y=roi_ymax,
            s="Region of interest\n(ROI)",
            fontsize=8,
            va="top",
            ha="right",
        )
    else:
        ax.text(
            x=roi_xmin,
            y=roi_ymin,
            s="Region of interest\n(ROI)",
            fontsize=8,
            va="top",
            ha="right",
        )

    handles = [
        Line2D([0], [0], color='k', linestyle='--', lw=1, label='Equity'),
        Line2D([0], [0], color='k', linestyle=':', lw=0.8, label='Error bounds'),
        # Patch(facecolor='lightblue', alpha=0.2, label='ROI')
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)


def normalize_name(name: str) -> str:
    if "_mepoml" in name:
        # Keep everything up to `_mepoml`
        return name.split("_mepoml")[0] + "_mepoml"
    else:
        # Add `_mepoml` if not present
        return name + "_mepoml"



# ------------------ LOAD DATA ------------------
co2_uptake = np.load(PRED_PATHS["co2_uptake"], allow_pickle=True)
kh_co2 = np.load(PRED_PATHS["kh_co2"], allow_pickle=True)
qst_co2 = np.load(PRED_PATHS["qst_co2"], allow_pickle=True)

df_pred = pd.DataFrame({
    "name": qst_co2["sid"],
    "qst_co2_pred": qst_co2["pred"],
    "kh_co2_pred": kh_co2["pred"],
    "co2_uptake_pred": co2_uptake["pred"],
})

gt = pd.read_csv(GT_CSV)
gt['name'] = gt['name'].apply(normalize_name)
df_pred['name'] = df_pred['name'].apply(normalize_name)


# Check for ID match before merging
gt_ids = set(gt['name'])
pred_ids = set(df_pred['name'])
common_ids = gt_ids & pred_ids
missing_in_pred = gt_ids - pred_ids
missing_in_gt = pred_ids - gt_ids

if len(common_ids) == 0:
    raise ValueError(
        "Couldn't find any matched IDs between predictions and ground-truth. "
        "Make sure you defined the correct path to the ground-truth (i.e., GT_CSV)."
    )

total_gt = len(gt_ids)
total_pred = len(pred_ids)

warnings.warn(
    f"ID overlap stats:\n"
    f"- Total GT IDs: {total_gt}\n"
    f"- Total Prediction IDs: {total_pred}\n"
    f"- Missing in predictions: {len(missing_in_pred)}\n"
    f"- Missing in ground truth: {len(missing_in_gt)}"
)

merged = pd.merge(gt, df_pred, on="name", how="inner")


# ------------------ NORMALIZATION ------------------
# DB1_STATS = {
#     "qst_co2": [-26.125842971948742, 6.8261558432098575],
#     "kh_co2": [-9.79689810,  1.38127887],
#     "co2_uptake": [-6.10493189,  1.38102965],
# }

# DB2_STATS = {
#     "qst_co2": [-21.329700577143377, 5.267456123165425],
#     "kh_co2": [-10.40316590478883,  0.9921236995164925],
#     "co2_uptake": [-6.7006051469045635, 1.0064174393812682],
# }

# merged['qst_co2_pred_denorm'] = merged['qst_co2_pred']
# merged['kh_co2_pred_denorm'] = merged['kh_co2_pred']
# merged['co2_uptake_pred_denorm'] = merged['co2_uptake_pred']


# ------------------ PLOT ------------------
fig, axes = plt.subplots(1, 3, figsize=(8, 3))

plot_equity_with_band(
    axes[0], 
    merged['qst_gt'],
    merged['qst_co2_pred'],
    key="qst_co2",
    title="Qst CO$_2$"
)


plot_equity_with_band(
    axes[1],
    np.log10(merged['co2_uptake_gt']),
    np.log10(merged['co2_uptake_pred']),
    title="CO$_2$ Uptake",
    log=True,
    key="co2_uptake"
)

plot_equity_with_band(
    axes[2],
    np.log10(merged['kh_co2_gt']),
    np.log10(merged['kh_co2_pred']),
    title="$K_H$ CO$_2$",
    log=True,
    key="kh_co2"
)

axes[0].text(-80, 0, s='AI-MOFs', fontsize=10, weight='bold', ha='left')

plt.tight_layout()
plt.savefig(f"ai_mofs_results_{batch_ID}_{normalization_task}.png", dpi=300)
plt.show()