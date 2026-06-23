import os, pickle, json, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

OUT = os.environ.get("CARD_WORK", "outputs") + "/curated_benchmark_v2"
SEED = 42

BACKBONES = [
    ("qwen3vl_2b", "hidden_states_qwen3vl_2b_curated_v2.npy", "hidden_states_qwen3vl_2b_meta.pkl", "Qwen3-VL-2B"),
    ("qwen3vl_4b", "hidden_states_qwen3vl_4b_curated_v2.npy", "hidden_states_qwen3vl_4b_meta.pkl", "Qwen3-VL-4B"),
    ("qwen3vl_8b", "hidden_states_qwen3vl_8b_curated_v2.npy", "hidden_states_qwen3vl_8b_meta.pkl", "Qwen3-VL-8B"),
    ("gemma3_12b", "hidden_states_gemma3_12b_curated_v2.npy", "hidden_states_gemma3_12b_meta.pkl", "Gemma-3-12B"),
    ("llava15_7b", "hidden_states_llava15_7b_curated_v2.npy", "hidden_states_llava15_7b_meta.pkl", "LLaVA-1.5-7B"),
]
CLASSES = ["crime", "hate", "misinfo", "privacy", "sexual", "violence"]

def refusal_feature(H, y_cat, classes):
    N, L, D = H.shape
    unsafe_mask = np.isin(y_cat, classes)
    safe_mask = (y_cat == "safe")
    feat = np.zeros((N, L), dtype=np.float32)
    for l in range(L):
        Xl = H[:, l, :].astype(np.float32)
        mu_unsafe = Xl[unsafe_mask].mean(0)
        mu_safe = Xl[safe_mask].mean(0)
        v = mu_unsafe - mu_safe
        v_n = v / (np.linalg.norm(v) + 1e-8)
        X_n = Xl / (np.linalg.norm(Xl, axis=1, keepdims=True) + 1e-8)
        feat[:, l] = X_n @ v_n
    return feat

def binary_one_vs_rest(X, y_id, classes):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    per_cat = {}
    for cid, cname in enumerate(classes):
        y = (y_id == cid).astype(int)
        aurocs = []
        for tr, te in cv.split(X, y):
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(C=1, max_iter=3000,
                                     class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            p = lr.predict_proba(sc.transform(X[te]))[:, 1]
            aurocs.append(roc_auc_score(y[te], p))
        per_cat[cname] = {"mean": float(np.mean(aurocs)), "std": float(np.std(aurocs))}
    return {"per_cat": per_cat, "mean": float(np.mean([v["mean"] for v in per_cat.values()]))}

def six_way_acc(X, y_id, classes):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs = []; pred = np.zeros(len(X), dtype=int)
    for tr, te in cv.split(X, y_id):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        lda = LinearDiscriminantAnalysis(n_components=5).fit(Xtr, y_id[tr])
        svm = LinearSVC(C=0.1, max_iter=3000, class_weight="balanced").fit(
            lda.transform(Xtr), y_id[tr])
        pred[te] = svm.predict(lda.transform(Xte))
        accs.append(float((pred[te] == y_id[te]).mean()))
    per_cat = {c: float((pred[y_id == i] == i).mean()) for i, c in enumerate(classes)}
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "per_cat": per_cat}

def binary_safe_unsafe(feat, y_cat, classes):
    yb = np.isin(y_cat, classes).astype(int)
    X = feat
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aurocs = []
    for tr, te in cv.split(X, yb):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=1, max_iter=3000, class_weight="balanced").fit(
            sc.transform(X[tr]), yb[tr])
        p = lr.predict_proba(sc.transform(X[te]))[:, 1]
        aurocs.append(roc_auc_score(yb[te], p))
    return {"mean": float(np.mean(aurocs)), "std": float(np.std(aurocs))}

results = {}
for tag, hs, meta_pkl, name in BACKBONES:
    if not os.path.exists(os.path.join(OUT, hs)):
        print(f"\n--- {name} --- [skip] {hs} not found"); continue
    print(f"\n--- {name} ---")
    H = np.load(os.path.join(OUT, hs))
    meta = pickle.load(open(os.path.join(OUT, meta_pkl), "rb"))["meta"]
    y_cat = np.array([m["meta_category"] for m in meta])

    print(f"  H={H.shape}, computing 36-D refusal-cosine feature ...")
    feat_full = refusal_feature(H, y_cat, CLASSES)
    print(f"  feature shape: {feat_full.shape}")

    unsafe_mask = np.isin(y_cat, CLASSES)
    feat_u = feat_full[unsafe_mask]
    le = LabelEncoder(); y_id = le.fit_transform(y_cat[unsafe_mask])

    print("  [1/3] per-cat binary AUROC ...")
    ovr = binary_one_vs_rest(feat_u, y_id, CLASSES)
    print(f"    {ovr}")

    print("  [2/3] 6-way accuracy ...")
    kw = six_way_acc(feat_u, y_id, CLASSES)
    print(f"    mean={kw['mean']*100:.2f}%  per_cat={kw['per_cat']}")

    print("  [3/3] binary safe/unsafe (2400) ...")
    bin_full = binary_safe_unsafe(feat_full, y_cat, CLASSES)
    print(f"    AUROC={bin_full['mean']:.4f}")

    results[tag] = {
        "name": name,
        "binary_one_vs_rest": ovr,
        "six_way_acc": kw["mean"],
        "six_way_std": kw["std"],
        "per_cat_6way": kw["per_cat"],
        "binary_safe_unsafe": bin_full,
    }

with open(os.path.join(OUT, "hiddendetect_baseline.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {os.path.join(OUT, 'hiddendetect_baseline.json')}")

print("\n=== SUMMARY ===")
print(f"{'Backbone':<18}{'Binary OVR':>14}{'6-way':>9}{'Binary Full':>14}")
for tag, r in results.items():
    print(f"{r['name']:<18}{r['binary_one_vs_rest']['mean']:>14.4f}"
          f"{r['six_way_acc']*100:>8.2f}%{r['binary_safe_unsafe']['mean']:>14.4f}")
