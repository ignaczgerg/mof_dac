import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import scienceplots


# ------------------ CONFIG: CHANGE THESE PATHS ONLY ------------------
BASE_PRED_DIR = "/ibex/project/c2261/dac_iclr/finetune/lightning_logs/kkfq442g/aramco_dac_finetune/kkfq442g/checkpoints/predictions_external/external_eval/"
normalization_task = "db_1"  # choose "db_1" or "db_2"

BATCHES = {
    "batch1": {
        "pred_subdir": "test_ai-mof-batch1",
        "gt_csv": "../evaluate_results/ai-generated_B1_gt.csv",
        "color": "#1f77b4",  # blue
    },
    "batch3": {
        "pred_subdir": "test_ai-mof-batch3",
        "gt_csv": "../evaluate_results/ai-generated_B3_gt.csv",
        "color": "#ff7f0e",  # orange
    },
}

# ------------------ STYLE ------------------
plt.style.use(['science', 'no-latex'])


# ------------------ FUNCTIONS ------------------
def smape(x, y, eps=1e-8):
    x, y = np.array(x), np.array(y)
    denominator = (np.abs(x) + np.abs(y)) / 2.0
    return np.mean(np.abs(y - x) / (denominator + eps)) * 100

def normalize_name(name: str) -> str:
    if "_mepoml" in name:
        return name.split("_mepoml")[0] + "_mepoml"
    else:
        return name + "_mepoml"

def plot_equity_with_band(ax, x, y, title, log=False, key=None, batch_indices=None, batch_colors=None):
    REGION_OF_INTEREST = {
        'qst_co2': [-30, -60],        
        'kh_co2':  [-3.5, 0],        
        'co2_uptake': [-2, 1],        
        'qst_h2o': [-40, -80],        
        'kh_h2o':  [-2, 1],           
        'selectivity_co2_h2o': [0, 3],
    }

    # ---- Scatter logic with multiple colors ----
    if batch_indices is None or batch_colors is None:
        ax.scatter(x, y, s=0.5)  # single color (default)
    else:
        for idx, color in zip(batch_indices, batch_colors):
            ax.scatter(x[idx], y[idx], s=0.5, color=color)

    r2 = r2_score(x, y)

    # minv, maxv = x.min(), x.max()
    # ax.plot([minv, maxv], [minv, maxv], 'k--', lw=1, label="Equity line")
    
    if key in REGION_OF_INTEREST:
        roi_xmin, roi_xmax = REGION_OF_INTEREST[key]
        roi_ymin, roi_ymax = REGION_OF_INTEREST[key]

        # Extend range to cover both data and ROI
        minv = min(x.min(), y.min(), roi_xmin, roi_ymin)
        maxv = max(x.max(), y.max(), roi_xmax, roi_ymax)
        if key == "co2_uptake":
            maxv = maxv - 0.8
        elif key == "kh_co2":
            maxv = maxv - 2
    else:
        minv = min(x.min(), y.min())
        maxv = max(x.max(), y.max())


    # Now plot the equity line across the full visible range
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

    ax.text(0.05, 0.88, f"Relative Error: {smape_error:.1f}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=7)

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
        ax.text(0.05, 0.95, f"$R^2$ (log): {r2:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=7)
    else:
        ax.text(0.05, 0.95, f"$R^2$ (norm): {r2:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=7)

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

        # ax.text(0.05, 0.88,
        #         f"$R^2$ (ROI): {r2_roi:.2f}" if in_roi >= 2 else "$R^2$ (ROI): -",
        #         transform=ax.transAxes, va="top", ha="left", fontsize=7)
        # ax.text(0.05, 0.81,
        #         f"In ROI: {in_roi/len(y):.0%}",
        #         transform=ax.transAxes, va="top", ha="left", fontsize=7)

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
            fontsize=7,
            va="top",
            ha="right",
        )
    else:
        ax.text(
            x=roi_xmax,
            y=roi_ymax,
            s="Region of interest\n(ROI)",
            fontsize=7,
            va="bottom",
            ha="left",
        )

    handles = [
        Line2D([0], [0], color='k', linestyle='--', lw=1, label='Equity'),
        Line2D([0], [0], color='k', linestyle=':', lw=0.8, label='Error bounds'),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7)



# ------------------ LOAD + CHECK + MERGE PER BATCH ------------------
def load_and_merge_one_batch(pred_subdir: str, gt_csv: str):
    pred_paths = {
        name: f"{BASE_PRED_DIR}/{pred_subdir}/{name}_{normalization_task}.npz"
        for name in ["co2_uptake", "kh_co2", "qst_co2"]
    }

    co2_uptake = np.load(pred_paths["co2_uptake"], allow_pickle=True)
    kh_co2 = np.load(pred_paths["kh_co2"], allow_pickle=True)
    qst_co2 = np.load(pred_paths["qst_co2"], allow_pickle=True)

    df_pred = pd.DataFrame({
        "name": qst_co2["sid"],
        "qst_co2_pred": qst_co2["pred"],
        "kh_co2_pred": kh_co2["pred"],
        "co2_uptake_pred": co2_uptake["pred"],
    })

    gt = pd.read_csv(gt_csv)
    gt['name'] = gt['name'].apply(normalize_name)
    df_pred['name'] = df_pred['name'].apply(normalize_name)

    # ---------- EXACT ID-MATCH LOGIC (kept verbatim) ----------
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
    # ----------------------------------------------------------

    merged = pd.merge(gt, df_pred, on="name", how="inner")
    return merged

# ------------------ LOAD DATA ------------------
# Load both batches into merged_per_batch
merged_per_batch = {}
for batch_id, info in BATCHES.items():
    merged_per_batch[batch_id] = load_and_merge_one_batch(
        pred_subdir=info["pred_subdir"],
        gt_csv=info["gt_csv"],
    )

# ------------------ PREPARE DATA FOR PLOTTING ------------------
# Concatenate all points for each target
def concat_for_target(target):
    return np.concatenate([merged_per_batch[batch_id][target] for batch_id in BATCHES])

qst_x_all = concat_for_target("qst_gt")
qst_y_all = concat_for_target("qst_co2_pred")
uptake_x_all = np.concatenate([np.log10(merged_per_batch[batch_id]['co2_uptake_gt']) for batch_id in BATCHES])
uptake_y_all = np.concatenate([np.log10(merged_per_batch[batch_id]['co2_uptake_pred']) for batch_id in BATCHES])
kh_x_all = np.concatenate([np.log10(merged_per_batch[batch_id]['kh_co2_gt']) for batch_id in BATCHES])
kh_y_all = np.concatenate([np.log10(merged_per_batch[batch_id]['kh_co2_pred']) for batch_id in BATCHES])


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

# qst_y_all = (qst_y_all * DB2_STATS['qst_co2'][1]) + DB2_STATS['qst_co2'][0]
# qst_y_all = (qst_y_all - DB1_STATS['qst_co2'][0]) / DB1_STATS['qst_co2'][1]


# Batch indices & colors for coloring
batch_indices = []
batch_colors = []
offset = 0
for batch_id, info in BATCHES.items():
    n = len(merged_per_batch[batch_id])
    batch_indices.append(np.arange(offset, offset+n))
    batch_colors.append(info["color"])
    offset += n

# ------------------ PLOT EVERYTHING TOGETHER ------------------
fig, axes = plt.subplots(1, 3, figsize=(8, 3))

# Qst CO2
plot_equity_with_band(
    axes[0], qst_x_all, qst_y_all,
    title="Qst CO$_2$",
    key="qst_co2",
    batch_indices=batch_indices,
    batch_colors=batch_colors
)

# CO2 Uptake
plot_equity_with_band(
    axes[1], uptake_x_all, uptake_y_all,
    title="CO$_2$ Uptake",
    log=True,
    key="co2_uptake",
    batch_indices=batch_indices,
    batch_colors=batch_colors
)
# --- Add KAUST-7 point and legend inside ax[1] ---
axes[1].scatter(x=[0.09], y=[0.09], s=50, color='red', marker='*', zorder=5)
# axes[1].legend(loc="lower right", fontsize=7)


# KH CO2
plot_equity_with_band(
    axes[2], kh_x_all, kh_y_all,
    title="$K_H$ CO$_2$",
    log=True,
    key="kh_co2",
    batch_indices=batch_indices,
    batch_colors=batch_colors
)

# ------------------ ADD LEGEND ------------------
legend_handles = [
    Line2D([0], [0], marker='o', linestyle='None', markersize=5,
           color=info["color"], label=f"Batch {bid[-1]}")
    for bid, info in BATCHES.items()
]
# Add KAUST-7 star handle
legend_handles.append(Line2D([0], [0], marker='*', color='red', linestyle='None',
                             markersize=7, label='KAUST-7'))

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(BATCHES)+1,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.05)
)

plt.tight_layout()
plt.savefig(f"ai_mofs_results_batch1_batch3_{normalization_task}.png", dpi=300)
plt.show()


# merged['qst_co2_pred_denorm'] = merged['qst_co2_pred']
# merged['kh_co2_pred_denorm'] = merged['kh_co2_pred']
# merged['co2_uptake_pred_denorm'] = merged['co2_uptake_pred']