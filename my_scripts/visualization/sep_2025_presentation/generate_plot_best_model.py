import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import scienceplots
import pandas as pd



# ---------- STYLE (match first plot) ----------
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

# Map labels to consistent colors
custom_colors = {
    "January": colors[1],
    "July": colors[2],
    "Now": colors[0],
}

# ---------- DATA ----------
raw_smape = {
    # "qst_co2":     {"January": 12.6, "July": 9.2, "Now": 6.7},
    # "kh_co2":      {"January": 63.2, "July": 43.5, "Now": 31.6},
    "co2_uptake":  {"January": 63.6, "July": 43.5, "Now": 32.1},
}

label_order = ["January", "July", "Now"]
metric_rename = {
    # "qst_co2":    r"CO$_2$ Qst",
    # "kh_co2":     r"CO$_2$ KH",
    "co2_uptake": r"CO$_2$ Uptake",
}
# metric_display_order = ["qst_co2", "kh_co2", "co2_uptake"]
metric_display_order = ["co2_uptake"]

# Flatten to long dataframe
rows = []
for metric, d in raw_smape.items():
    for label, val in d.items():
        rows.append({"metric": metric, "label_display": label, "value": float(val)})
smape_df = pd.DataFrame(rows)

# ---------- ANNOTATION HELPER ----------
def annotate_bars(ax, fmt="{:.1f}%", dy=None, fontsize=20):
    ymax = ax.get_ylim()[1]
    if dy is None:
        dy = 0.018 * ymax
    for p in ax.patches:
        h = p.get_height()
        if h is None:
            continue
        x = p.get_x() + p.get_width() / 2
        ax.text(x, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, clip_on=True)

# ---------- PLOT ----------
fig_smape, axs_smape = plt.subplots(1, len(metric_display_order),
                                    figsize=(5.6 * len(metric_display_order), 5.6))

if len(metric_display_order) == 1:
    axs_smape = [axs_smape]

for ax, metric in zip(axs_smape, metric_display_order):
    subset = smape_df[smape_df["metric"] == metric].copy()
    subset["label_display"] = pd.Categorical(subset["label_display"],
                                             categories=label_order, ordered=True)
    subset.sort_values("label_display", inplace=True)

    max_val = subset["value"].max()
    ymin = 0
    ymax = max_val * 1.15 if max_val > 0 else 1.0

    sns.barplot(
        data=subset,
        x="label_display",
        y="value",
        hue="label_display",
        palette=[custom_colors[l] for l in subset["label_display"].unique()],
        ax=ax,
        legend=False,
        edgecolor="black",
        linewidth=1.0,
        width=0.35,   # <<--- wider bars
    )

    ax.set_title(metric_rename[metric])
    ax.set_ylabel("SMAPE (%)")
    ax.set_xlabel("")
    ax.set_ylim([ymin, ymax])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, linestyle="--", linewidth=0.6)

    annotate_bars(ax, fmt="{:.1f}%", fontsize=20)

fig_smape.tight_layout()
plt.savefig("current_best_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
