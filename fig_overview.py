import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
FIG = os.environ.get("CARD_WORK", "outputs") + "/figures/fig_overview.png"
os.makedirs(os.path.dirname(FIG), exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
C = {"in": "#e8f0fe", "vlm": "#dfe9d8", "feat": "#fff3cd", "head": "#fde2e2", "out": "#e7e0f5"}
def box(x, y, w, h, text, c, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                fc=c, ec="#555", lw=1.2))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                lw=1.5, color="#444"))

box(0.2, 1.8, 1.8, 1.0, "Input\n(image, text)", C["in"], 9, True)
box(2.5, 1.6, 2.0, 1.4, "Frozen VLM\nsingle prefill pass\n(no gradient)", C["vlm"], 9, True)
box(5.0, 1.6, 2.1, 1.4, "Per-layer hidden\nstates $\\{h_l\\}_{l=1}^{L}$\n$\\to$ PCA-whiten +\nCV-selected layer", C["feat"], 8.5, True)
box(7.7, 2.7, 2.2, 1.1, "Binary head\naggregated SafetyScore\n$>\\tau$ ?", C["head"], 8.5, True)
box(7.7, 0.7, 2.2, 1.1, "$K$-way head\nsmall MLP (CE+LS)\ngradient-free", C["head"], 8.5, True)
box(10.2, 2.75, 1.7, 1.0, "SAFE /\nUNSAFE\n(intercept)", C["out"], 9, True)
box(10.2, 0.75, 1.7, 1.0, "category:\nviolence/sexual/\nhate/crime/\nprivacy/misinfo", C["out"], 7.5, True)
arrow(2.0, 2.3, 2.5, 2.3); arrow(4.5, 2.3, 5.0, 2.3)
arrow(7.1, 2.5, 7.7, 3.05); arrow(7.1, 2.1, 7.7, 1.25)
arrow(9.9, 3.25, 10.2, 3.25); arrow(9.9, 1.25, 10.2, 1.25)
ax.text(6.0, 4.25, "CARD: gradient-free, $K$-way-capable prefill-time safety detector",
        ha="center", fontsize=12, fontweight="bold")
ax.text(6.0, 0.25, "VLM frozen (no backprop) $\\cdot$ $\\sim$100 calibration samples/class $\\cdot$ "
        "$<$2\\,ms beyond prefill", ha="center", fontsize=8.5, style="italic", color="#333")
plt.tight_layout()
plt.savefig(FIG, dpi=160, bbox_inches="tight"); print("saved", FIG)
