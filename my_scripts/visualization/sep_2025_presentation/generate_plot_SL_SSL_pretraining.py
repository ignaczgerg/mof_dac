import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import scienceplots

# ---------- STYLE ----------
plt.style.use(['science', 'no-latex'])
colors = sns.color_palette("Set2", 8)  # ensure enough indices
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 24,
})

# Colors
custom_colors = {"OC20": colors[2]}   # blue-ish
training_type_colors = {
    "Supervised": custom_colors["OC20"],
    "Autoreg":    colors[4],          # distinct color for Autoreg (purple-ish)
}

# ---------- RAW DATA (SMAPE %) ----------
raw_smape = {
    "qst_co2":    {"OC20": {"Supervised": 12.0, "Autoreg": 11.4}},
    "kh_co2":     {"OC20": {"Supervised": 62.1, "Autoreg": 56.8}},
    "co2_uptake": {"OC20": {"Supervised": 62.4, "Autoreg": 57.4}},
}

metric_rename = {
    "qst_co2":    r"CO$_2$ Qst",
    "kh_co2":     r"CO$_2$ KH",
    "co2_uptake": r"CO$_2$ Uptake",
}
metric_display_order = ["qst_co2", "kh_co2", "co2_uptake"]
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
    """Write the numeric value above vertical bars."""
    ymax = ax.get_ylim()[1]
    if dy is None:
        dy = 0.018 * ymax
    for r in rects:
        h = r.get_height()
        ax.text(
            r.get_x() + r.get_width()/2,
            h + dy,
            fmt.format(h),
            ha="center", va="bottom", fontsize=fontsize
        )

# ---------- DRAW ----------
fig, axs = plt.subplots(1, len(metric_display_order), figsize=(5.6 * len(metric_display_order), 5.6), sharey=False)

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

        # center tick between the two bars
        xtick_positions.append(current_pos + bar_spacing / 2)
        xtick_labels.append("Pretraining on OC20")  # <-- updated x-axis tick label
        current_pos += group_spacing

    metric_vals = smape_df[smape_df["metric"] == metric]["value"]
    ax.set_ylim([0, metric_vals.max() * 1.14])  # a bit more headroom for larger labels

    ax.set_title(metric_rename[metric])
    ax.set_ylabel("SMAPE (%)")
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.tick_params(axis="x", pad=6)
    ax.grid(True, linestyle="--", linewidth=0.6)

    # Annotate after limits are set
    annotate_bars(ax, rects, fmt="{:.1f}%", fontsize=20)

# Legend
legend_patches = [
    Patch(facecolor=training_type_colors["Supervised"], label='Supervised', edgecolor="black"),
    Patch(facecolor=training_type_colors["Autoreg"],    label='Autoregressive', edgecolor="black"),
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=2,
    frameon=False
)

fig.suptitle("OC20: Supervised vs Autoregressive Pretraining (SMAPE)", fontsize=30, y=0.98)
fig.tight_layout()
fig.savefig("oc20_supervised_vs_autoreg_smape.png", dpi=300, bbox_inches="tight")
print("Saved: oc20_supervised_vs_autoreg_smape.png")
