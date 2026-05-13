# OC20 training cost (V100 GPU-days): horizontal bars, no legend,
# added vertical spacing for title and x-axis label

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import scienceplots

# ---------- STYLE ----------
plt.style.use(['science', 'no-latex'])
colors = sns.color_palette("Set2", 8)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
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
y_pos  = range(len(names))

# Colors
c_supervised = colors[2]   # blue-ish
c_autoreg    = colors[4]   # distinct

# ---------- PLOT ----------
fig, ax = plt.subplots(figsize=(9.5, 4.8))

bars = ax.barh(
    y_pos, values, height=0.45,
    color=[c_supervised, c_autoreg],
    edgecolor="black"
)

# Title with extra spacing; x-label with extra spacing
ax.set_title("Pretraining on OC20", pad=20)                 # <-- extra top space
ax.set_xlabel("V100 GPU-days", labelpad=14)                 # <-- extra bottom space

ax.set_yticks(list(y_pos))
ax.set_yticklabels(names)
ax.grid(True, axis="x", linestyle="--", linewidth=0.6)

# Headroom + numeric labels on bar ends
max_v = max(values)
ax.set_xlim(0, max_v * 1.25)
ax.bar_label(bars, labels=[str(v) for v in values], padding=6, fontsize=18)

# No legend (removed)

# Add overall top/bottom margins for more breathing room
fig.subplots_adjust(top=0.86, bottom=0.22)

# Save
plt.savefig("oc20_gpu_days.png", dpi=300, bbox_inches="tight")
print("Saved: oc20_gpu_days.png")
