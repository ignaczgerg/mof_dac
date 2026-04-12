import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import scienceplots

# ---------- STYLE ----------
plt.style.use(['science', 'no-latex'])
colors = sns.color_palette("Set2")
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

custom_colors = {
    "Scratch": colors[3],  # pink
    "OC20":    colors[2],  # blue
    "JMP":     colors[1],  # orange
    "ODAC":    colors[0],  # green
}

# ---------- RAW DATA (SMAPE %) ----------
# Values approximated from the attached figure
raw_smape = {
    "qst_co2":     {"Scratch": 12.6, "OC20": 12.0, "JMP": 11.5, "ODAC": 10.2},
    "kh_co2":      {"Scratch": 63.2, "OC20": 62.1, "JMP": 63.4, "ODAC": 52.5},
    "co2_uptake":  {"Scratch": 63.6, "OC20": 62.4, "JMP": 62.2, "ODAC": 53.0},
}

label_order = ["Scratch", "OC20", "JMP", "ODAC"]
metric_rename = {
    "qst_co2":    r"CO$_2$ Qst",
    "kh_co2":     r"CO$_2$ KH",
    "co2_uptake": r"CO$_2$ Uptake",
}
metric_display_order = ["qst_co2", "kh_co2", "co2_uptake"]

# Build long DataFrame from raw dict
rows = []
for metric, d in raw_smape.items():
    for label, val in d.items():
        rows.append({"metric": metric, "label_display": label, "value": float(val)})
smape_df = pd.DataFrame(rows)

# ---------- PLOT (SMAPE) ----------
fig_smape, axs_smape = plt.subplots(1, len(metric_display_order), figsize=(5 * len(metric_display_order), 5))

for ax, metric in zip(axs_smape, metric_display_order):
    subset = smape_df[smape_df["metric"] == metric].copy()
    subset["label_display"] = pd.Categorical(subset["label_display"],
                                             categories=label_order, ordered=True)
    subset.sort_values("label_display", inplace=True)

    # Y range with a small headroom and trimmed bottom margin
    min_val = subset["value"].min()
    max_val = subset["value"].max()
    margin = (max_val - min_val) * 0.10 if max_val > min_val else 1.0
    ymin = 0
    ymax = max_val * 1.02

    sns.barplot(
        data=subset,
        x="label_display",
        y="value",
        hue="label_display",
        palette=[custom_colors[l] for l in subset["label_display"].unique()],
        ax=ax,
        legend=False,
        edgecolor="black",
        width=0.6
    )
    ax.set_title(metric_rename[metric])
    ax.set_ylabel("SMAPE (%)")
    ax.set_xlabel("")
    ax.set_ylim([ymin, ymax])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, linestyle="--", linewidth=0.5)

fig_smape.suptitle("Impact of Pretraining Sources on Downstream Performance (SMAPE)", fontsize=24, y=0.95)
fig_smape.tight_layout()
fig_smape.savefig("pretraining_sources_smape1.png", dpi=300, bbox_inches="tight")
print("Saved: pretraining_sources_smape1.png")
