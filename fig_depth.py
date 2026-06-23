import os, sys, pickle, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

OUT = os.environ.get("CARD_WORK", "outputs") + "/curated_benchmark_v2"
FIGDIR = os.environ.get("CARD_WORK", "outputs") + "/figures"
os.makedirs(FIGDIR, exist_ok=True)
CLASSES = ["crime", "hate", "misinfo", "privacy", "sexual", "violence"]
tag = sys.argv[1] if len(sys.argv) > 1 else "qwen3vl_8b"
npy = {"qwen3vl_8b": "hidden_states_qwen3vl_8b_curated_v2.npy",
       "qwen3vl_4b": "hidden_states_qwen3vl_4b_curated_v2.npy"}[tag]
mpk = npy.replace("_curated_v2.npy", "_meta.pkl")
name = {"qwen3vl_8b": "Qwen3-VL-8B", "qwen3vl_4b": "Qwen3-VL-4B"}[tag]

H = np.load(os.path.join(OUT, npy))
meta = pickle.load(open(os.path.join(OUT, mpk), "rb"))["meta"]
y_cat = np.array([m["meta_category"] for m in meta])
um = np.isin(y_cat, CLASSES)
H_u = H[um].astype(np.float32)
y_id = LabelEncoder().fit(CLASSES).transform(y_cat[um])
N = H_u.shape[1]; w = 4
print(f"{name}: layers={N}", flush=True)

fig, ax = plt.subplots(figsize=(9, 4.5))
peak_depths = []
for cname in ["violence", "sexual", "privacy", "crime", "hate", "misinfo"]:
    cid = CLASSES.index(cname)
    y = (y_id == cid).astype(int)
    curve = []
    for ls in range(N - w + 1):
        Xw = H_u[:, ls:ls + w, :].mean(1)
        v = Xw[y == 1].mean(0) - Xw[y == 0].mean(0); v /= (np.linalg.norm(v) + 1e-8)
        a = roc_auc_score(y, Xw @ v); a = max(a, 1 - a); curve.append(a)
    xs = [x / max(1, N - 1) * 100 for x in range(len(curve))]
    ax.plot(xs, curve, label=cname, linewidth=1.6, alpha=0.9)
    peak_depths.append(xs[int(np.argmax(curve))])
ax.set_xlabel("Window start position (% of network depth)")
ax.set_ylabel("Fisher-direction AUROC")
ax.legend(fontsize=9, loc="lower right", ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
outp = os.path.join(FIGDIR, "layer_window_auroc.png")
plt.savefig(outp, dpi=150, bbox_inches="tight")
print(f"Saved {outp}", flush=True)
print(f"per-category peak window-start depths (%): {[round(p,1) for p in peak_depths]}", flush=True)
print(f"mean peak depth: {np.mean(peak_depths):.1f}%", flush=True)
