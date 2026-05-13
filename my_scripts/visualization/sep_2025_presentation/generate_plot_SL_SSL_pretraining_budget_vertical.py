# OC20 training cost (V100 GPU-days): vertical bars, no legend
# Match SMAPE figure size (5.6 x 5.6) and bar width (0.1)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import scienceplots
from matplotlib.ticker import FuncFormatter

# ---------- STYLE ----------
plt.style.use(['science', 'no-latex'])
colors = sns.color_palette("Set2", 8)
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

# ---------- DATA ----------
gpu_days = {"Supervised": 705, "Autoregressive": 6}
names  = ["Supervised", "Autoregressive"]
values = [gpu_days[n] for n in names]

# Colors (consistent with your SMAPE palette)
c_supervised = colors[2]
c_autoreg    = colors[4]

# ---------- BAR LAYOUT (match SMAPE) ----------
bar_width   = 0.1
bar_spacing = 0.2

x0 = 0.0
x_pos = [x0, x0 + bar_spacing]                 # two bars in one group
xtick_positions = [x0 + bar_spacing / 2]       # centered tick
xtick_labels    = [""]

def annotate_bars(ax, rects, fmt="{}", dy=None, fontsize=16):
    ymax = ax.get_ylim()[1]
    if dy is None:
        dy = 0.018 * ymax
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize)

# ---------- PLOT ----------
fig, ax = plt.subplots(figsize=(5.6, 5.6))     # <-- same size as single SMAPE panel

rects = []
rects.append(ax.bar(x_pos[0], values[0], bar_width, color=c_supervised, edgecolor="black")[0])
rects.append(ax.bar(x_pos[1], values[1], bar_width, color=c_autoreg,    edgecolor="black")[0])

ax.set_title("Cost of Pretraining on OC20", pad=30)
ax.set_ylabel("V100 GPU-days")

ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels)
ax.grid(True, axis="y", linestyle="--", linewidth=0.6)

max_v = max(values)
ax.set_ylim(0, max_v * 1.14)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: "" if int(round(x)) == 800 else f"{int(x)}")
)

# Numeric labels on top of bars
annotate_bars(ax, rects, fmt="{}", fontsize=20)

# ---------- LEGEND ----------
legend_handles = [
    Patch(facecolor=c_supervised, edgecolor="black", label="Supervised (MIT, FAIR Meta)"),
    Patch(facecolor=c_autoreg,    edgecolor="black", label="Autoregressive (KAUST)"),
]

fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.08),
           ncol=2, frameon=False)
# No legend
fig.tight_layout()
fig.subplots_adjust(top=0.86) 
plt.savefig("oc20_gpu_days.png", dpi=300, bbox_inches="tight", pad_inches=0.25)
print("Saved: oc20_gpu_days.png")
