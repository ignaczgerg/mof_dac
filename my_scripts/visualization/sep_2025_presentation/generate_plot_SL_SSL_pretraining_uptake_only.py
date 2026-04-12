import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import scienceplots

# ---------- STYLE ----------
plt.style.use(['science', 'no-latex'])
colors = sns.color_palette("Set2", 8)
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 24,
})

# Colors
custom_colors = {"OC20": colors[2]}
training_type_colors = {
    "Supervised": custom_colors["OC20"],
    "Autoreg":    colors[4],
}

# ---------- RAW DATA (SMAPE %) ----------
raw_smape = {
    "co2_uptake": {"OC20": {"Supervised": 62.4, "Autoreg": 57.4}},
}

metric_rename = {"co2_uptake": r"CO$_2$ Uptake"}
metric_display_order = ["co2_uptake"]
groups = ["OC20"]

# Build long DF
rows = []
for metric, gdict in raw_smape.items():
    for group, tdict in gdict.items():
        for ttype, val in tdict.items():
            rows.append({
                "metric": metric,
                "base_label": group,
                "training_type": "Autoreg" if ttype.lower().startswith("autoreg") else "Supervised",
                "value": float(val),
            })
smape_df = pd.DataFrame(rows)

# ---------- BAR LAYOUT ----------
bar_width   = 0.1
bar_spacing = 0.2
group_spacing = 0.7

def annotate_bars(ax, rects, fmt="{:.1f}", dy=None, fontsize=16):
    ymax = ax.get_ylim()[1]
    if dy is None:
        dy = 0.018 * ymax
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize)

# ---------- DRAW ----------
n_panels = len(metric_display_order)
fig, axs = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 5.6), sharey=False)
if n_panels == 1:
    axs = [axs]  # make iterable

for ax, metric in zip(axs, metric_display_order):
    rects = []
    xtick_positions, xtick_labels = [], []
    current_pos = 0.0

    for group in groups:
        group_data = smape_df[(smape_df["base_label"] == group) & (smape_df["metric"] == metric)]

        for j, ttype in enumerate(["Supervised", "Autoreg"]):
            entry = group_data[group_data["training_type"] == ttype]
            if not entry.empty:
                val = float(entry["value"].values[0])
                xpos = current_pos + j * bar_spacing
                rect = ax.bar(
                    xpos, val, bar_width,
                    color=training_type_colors[ttype],
                    edgecolor="black"
                )[0]
                rects.append(rect)

        xtick_positions.append(current_pos + bar_spacing / 2)
        xtick_labels.append("Pretraining on OC20, Finetuning on DB1")
        current_pos += group_spacing

    metric_vals = smape_df[smape_df["metric"] == metric]["value"]
    ax.set_ylim([0, metric_vals.max() * 1.14])

    ax.set_title(metric_rename[metric], pad=30)
    ax.set_ylabel("SMAPE (%)")
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.tick_params(axis="x", pad=15)
    ax.grid(True, linestyle="--", linewidth=0.6)

    annotate_bars(ax, rects, fmt="{:.1f}%", fontsize=20)

# Legend
# legend_patches = [
#     Patch(facecolor=training_type_colors["Supervised"], label='Supervised', edgecolor="black"),
#     Patch(facecolor=training_type_colors["Autoreg"],    label='Autoregressive', edgecolor="black"),
# ]
# fig.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.08),
#            ncol=2, frameon=False)

# fig.suptitle("OC20: Supervised vs Autoregressive Pretraining (SMAPE)", fontsize=30, y=0.98)
fig.tight_layout()
fig.subplots_adjust(top=0.86) 
fig.savefig("oc20_supervised_vs_autoreg_smape.png", dpi=300, bbox_inches="tight", pad_inches=0.25)
print("Saved: oc20_supervised_vs_autoreg_smape.png")
