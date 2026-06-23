import os, pickle, json, numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_mm_safety import (
    compute_dd_probe_layers, per_probe_score, find_window, bin_metrics,
    CURATED_OUT, EXT_OUT, RES_OUT, W, UNSAFE
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")

def load(model_tag):
    if model_tag == "qwen3vl_8b":
        cur = "hidden_states_qwen3vl_8b_curated_v2.npy"; cmeta = "hidden_states_qwen3vl_8b_meta.pkl"
    elif model_tag == "qwen3vl_4b":
        cur = "hidden_states_qwen3vl_4b_curated_v2.npy"; cmeta = "hidden_states_qwen3vl_4b_meta.pkl"
    else:
        cur = "hidden_states_qwen3vl_2b_curated_v2.npy"; cmeta = "hidden_states_qwen3vl_2b_meta.pkl"
    H_cur = np.load(os.path.join(CURATED_OUT, cur))
    m = pickle.load(open(os.path.join(CURATED_OUT, cmeta), "rb"))["meta"]
    y_cur = np.array([mm["meta_category"] for mm in m])
    ext = os.path.join(EXT_OUT, f"jailbreakv_hidden_states_{model_tag}.npy")
    emeta = os.path.join(EXT_OUT, f"jailbreakv_meta_{model_tag}.pkl")
    if not os.path.exists(ext): return None
    H_ext = np.load(ext)
    m_ext = pickle.load(open(emeta, "rb"))["meta"]
    keep_e = np.array(['ERROR' not in r for r in m_ext])
    H_ext = H_ext[keep_e]
    m_ext = [r for i, r in enumerate(m_ext) if keep_e[i]]
    attacks = np.array([r["attack_type"] for r in m_ext])
    return H_cur, y_cur, H_ext, attacks

def run(model_tag):
    print(f"\n=== {model_tag} JailBreakV-28K ===")
    d = load(model_tag)
    if d is None: print("  [skip]"); return None
    H_cur, y_cur, H_ext, attacks = d
    print(f"  curated={H_cur.shape}, jbv_ext={H_ext.shape}")
    y_cur_bin = np.isin(y_cur, UNSAFE).astype(int)
    safe_mask = (y_cur == "safe")
    H_safe = H_cur[safe_mask]
    H_test = np.concatenate([H_ext, H_safe], axis=0)
    y_test = np.concatenate([np.ones(len(H_ext)), np.zeros(len(H_safe))]).astype(int)
    attacks_test = np.concatenate([attacks, np.array(["_safe"] * len(H_safe))])
    print(f"  positives (JBV jailbreaks) = {len(H_ext)}, negatives (curated safe) = {len(H_safe)}")

    v_ref = compute_dd_probe_layers(H_cur.astype(np.float32), y_cur, "refusal")
    hd_cur = per_probe_score(H_cur.astype(np.float32), v_ref)
    hd_test = per_probe_score(H_test.astype(np.float32), v_ref)
    sc_hd = StandardScaler().fit(hd_cur)
    lr_hd = LogisticRegression(C=1, max_iter=3000, class_weight="balanced").fit(
        sc_hd.transform(hd_cur), y_cur_bin)
    hd_scores = lr_hd.predict_proba(sc_hd.transform(hd_test))[:, 1]

    probes = ["refusal"] + UNSAFE
    pf_cur = np.zeros((len(H_cur), len(probes)), dtype=np.float32)
    pf_test = np.zeros((len(H_test), len(probes)), dtype=np.float32)
    for pi, p in enumerate(probes):
        v = compute_dd_probe_layers(H_cur.astype(np.float32), y_cur, p)
        s_cur = per_probe_score(H_cur.astype(np.float32), v)
        s_test = per_probe_score(H_test.astype(np.float32), v)
        ls, _ = find_window(s_cur, y_cur_bin, s_cur.shape[1], W)
        tr, te = np.trapz(s_cur[:, ls:ls+W], axis=1), np.trapz(s_test[:, ls:ls+W], axis=1)
        mu, sig = tr.mean(), tr.std()
        pf_cur[:, pi] = (tr - mu) / (sig + 1e-8)
        pf_test[:, pi] = (te - mu) / (sig + 1e-8)
    sc = StandardScaler().fit(pf_cur)
    lr = LogisticRegression(C=1, max_iter=3000, class_weight="balanced").fit(sc.transform(pf_cur), y_cur_bin)
    card_scores = lr.predict_proba(sc.transform(pf_test))[:, 1]

    R = {"overall_HD": bin_metrics(y_test, hd_scores),
         "overall_CARD": bin_metrics(y_test, card_scores), "per_attack": {}}
    print(f"  OVERALL HD:  AUROC={R['overall_HD']['auroc']:.4f}  C@0.1%={R['overall_HD']['catch']['0.001']*100:.1f}%")
    print(f"  OVERALL CARD: AUROC={R['overall_CARD']['auroc']:.4f}  C@0.1%={R['overall_CARD']['catch']['0.001']*100:.1f}%")
    for a in sorted(set(attacks.tolist())):
        m = (attacks_test == a) | (attacks_test == "_safe")
        y = ((attacks_test == a)[m]).astype(int)
        if y.sum() < 20 or (len(y)-y.sum()) < 20: continue
        rh = bin_metrics(y, hd_scores[m]); rc = bin_metrics(y, card_scores[m])
        R["per_attack"][a] = {"n": int(y.sum()), "HD": rh, "CARD": rc}
        print(f"  {a:<20}  HD:{rh['auroc']:.4f}/{rh['catch']['0.001']*100:.1f}% | CARD:{rc['auroc']:.4f}/{rc['catch']['0.001']*100:.1f}%")
    return R

ALL = {}
for tag in ["qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b"]:
    r = run(tag)
    if r: ALL[tag] = r
os.makedirs(RES_OUT, exist_ok=True)
out = os.path.join(RES_OUT, "jailbreakv_external.json")
with open(out, "w") as f:
    json.dump(ALL, f, indent=2, default=str)
print(f"\nSaved: {out}")
