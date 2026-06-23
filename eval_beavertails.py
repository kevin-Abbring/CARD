import os, pickle, json, numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, roc_curve
import warnings; warnings.filterwarnings("ignore")

CURATED_OUT = os.environ.get("CARD_WORK", "outputs") + "/curated_benchmark_v2"
EXT_OUT = os.environ.get("CARD_WORK", "outputs") + "/external_eval"
RES_OUT = os.environ.get("CARD_WORK", "outputs") + "/result"
SEED = 42
W = 4

META_CATEGORY = {
    "violence_and_physical_harm": "violence", "horror_and_gore": "violence",
    "dangerous_behavior": "violence", "animal_abuse": "violence",
    "terrorism_or_extremism": "violence",
    "pornographic_content": "sexual", "sexual_crimes": "sexual",
    "insulting_and_harassing_behavior": "hate", "discriminatory_depictions": "hate",
    "privacy_invasion_and_surveillance": "privacy",
    "sensitive_information_in_key_areas": "privacy",
    "identity_misuse_and_impersonation": "privacy",
    "hacking_or_digital_crime": "crime", "financial_and_academic_fraud": "crime",
    "false_information": "misinfo", "deception_in_personal_relationships": "misinfo",
    "psychological_harm_and_manipulation": "misinfo",
    "environmental_damage": "_dist", "psychological_horror_and_dark_themes": "_dist",
    "harmful_health_content": "_dist",
}
UNSAFE6 = ["crime", "hate", "misinfo", "privacy", "sexual", "violence"]

def log(msg): print(msg, flush=True)

def compute_dd_probe_layers(H_tr, y_tr_cat, probe_name):
    N, L, D = H_tr.shape
    v = np.zeros((L, D), dtype=np.float32)
    if probe_name == "refusal":
        pos_mask = np.isin(y_tr_cat, UNSAFE6)
        neg_mask = ~pos_mask
    else:
        pos_mask = (y_tr_cat == probe_name)
        neg_mask = ~pos_mask
    for l in range(L):
        Xl = H_tr[:, l, :]
        pos, neg = Xl[pos_mask].mean(0), Xl[neg_mask].mean(0)
        vv = pos - neg; vv = vv / (np.linalg.norm(vv) + 1e-8)
        v[l] = vv
    return v

def per_probe_score(H, v_per_layer):
    N, L, D = H.shape
    scores = np.zeros((N, L), dtype=np.float32)
    for l in range(L):
        Xl = H[:, l, :].astype(np.float32)
        Xl_n = Xl / (np.linalg.norm(Xl, axis=1, keepdims=True) + 1e-8)
        scores[:, l] = Xl_n @ v_per_layer[l]
    return scores

def find_window(scores_tr, y_tr_bin, L, w=W):
    best = (0, 0.5)
    for ls in range(L - w + 1):
        integ = np.trapz(scores_tr[:, ls:ls+w], axis=1)
        try:
            a = roc_auc_score(y_tr_bin, integ)
            if a < 0.5: a = 1 - a
            if a > best[1]: best = (ls, a)
        except: pass
    return best

def per_cat_ref_layer(H_u, y_id):
    N = H_u.shape[1]
    out = {}
    for cid in range(6):
        y = (y_id == cid).astype(int)
        best = (0, 0.0)
        for l in range(N):
            Xl = H_u[:, l, :]
            pos, neg = Xl[y == 1].mean(0), Xl[y == 0].mean(0)
            v = pos - neg; v = v / (np.linalg.norm(v) + 1e-8)
            a = roc_auc_score(y, Xl @ v)
            if a < 0.5: a = 1 - a
            if a > best[1]: best = (l, a)
        out[cid] = best[0]
    return out

def bin_metrics(y_bin, scores):
    auroc = float(roc_auc_score(y_bin, scores))
    fpr, tpr, _ = roc_curve(y_bin, scores)
    catch = {}
    for t in [0.001, 0.005, 0.01, 0.05]:
        idx = np.searchsorted(fpr, t); idx = min(idx, len(fpr)-1)
        catch[str(t)] = float(tpr[idx])
    return {"auroc": auroc, "catch": catch}

def load_backbone_data(model_tag):
    if model_tag == "qwen3vl_8b":
        curated_hs = "hidden_states_qwen3vl_8b_curated_v2.npy"
        curated_meta = "hidden_states_qwen3vl_8b_meta.pkl"
    elif model_tag == "qwen3vl_4b":
        curated_hs = "hidden_states_qwen3vl_4b_curated_v2.npy"
        curated_meta = "hidden_states_qwen3vl_4b_meta.pkl"
    elif model_tag == "qwen3vl_2b":
        curated_hs = "hidden_states_qwen3vl_2b_curated_v2.npy"
        curated_meta = "hidden_states_qwen3vl_2b_meta.pkl"
    else:
        raise ValueError(model_tag)

    H_curated = np.load(os.path.join(CURATED_OUT, curated_hs))
    m_curated = pickle.load(open(os.path.join(CURATED_OUT, curated_meta), "rb"))["meta"]
    y_cur = np.array([m["meta_category"] for m in m_curated])
    q_cur = np.array([m.get("text", m.get("question", ""))[:200] for m in m_curated])

    btv_hs = os.path.join(EXT_OUT, f"btv_eval_hidden_states_{model_tag}.npy")
    btv_meta = os.path.join(EXT_OUT, f"btv_eval_meta_{model_tag}.pkl")
    if not (os.path.exists(btv_hs) and os.path.exists(btv_meta)):
        log(f"  [WARN] BeaverTails-V extraction not found for {model_tag} — skip")
        return None
    H_btv = np.load(btv_hs)
    m_btv = pickle.load(open(btv_meta, "rb"))["meta"]
    y_btv_meta = np.array([META_CATEGORY.get(m["category"], "_dist") for m in m_btv])
    y_btv_fine = np.array([m["category"] for m in m_btv])
    q_btv = np.array([m["question"][:200] for m in m_btv])

    curated_q_set = set(q_cur)
    btv_dedup_mask = np.array([q not in curated_q_set for q in q_btv])
    H_btv = H_btv[btv_dedup_mask]
    y_btv_meta = y_btv_meta[btv_dedup_mask]
    y_btv_fine = y_btv_fine[btv_dedup_mask]
    q_btv = q_btv[btv_dedup_mask]

    log(f"  [{model_tag}] curated={H_curated.shape}, BTV_eval={H_btv.shape} (after dedup)")
    return {
        "H_curated": H_curated, "y_curated": y_cur,
        "H_btv": H_btv, "y_btv_meta": y_btv_meta, "y_btv_fine": y_btv_fine,
    }

def test_t1_binary(data, model_tag):
    H_cur = data["H_curated"]; y_cur = data["y_curated"]
    H_btv = data["H_btv"]
    safe_mask = (y_cur == "safe")
    H_safe = H_cur[safe_mask]
    N_unsafe = len(H_btv); N_safe = len(H_safe)
    log(f"  T1: positives (BTV unsafe) = {N_unsafe}, negatives (curated safe) = {N_safe}")

    H_test = np.concatenate([H_btv, H_safe], axis=0)
    y_test = np.concatenate([np.ones(N_unsafe), np.zeros(N_safe)]).astype(int)

    y_cur_binary = np.isin(y_cur, UNSAFE6).astype(int)

    v_ref = compute_dd_probe_layers(H_cur.astype(np.float32), y_cur, "refusal")
    hd_feat_cur = per_probe_score(H_cur.astype(np.float32), v_ref)
    hd_feat_test = per_probe_score(H_test.astype(np.float32), v_ref)
    sc_hd = StandardScaler().fit(hd_feat_cur)
    lr_hd = LogisticRegression(C=1, max_iter=3000, class_weight="balanced").fit(
        sc_hd.transform(hd_feat_cur), y_cur_binary)
    hd_scores_test = lr_hd.predict_proba(sc_hd.transform(hd_feat_test))[:, 1]

    probes = ["refusal"] + UNSAFE6
    pf_cur = np.zeros((len(H_cur), len(probes)), dtype=np.float32)
    pf_test = np.zeros((len(H_test), len(probes)), dtype=np.float32)
    for pi, p in enumerate(probes):
        v = compute_dd_probe_layers(H_cur.astype(np.float32), y_cur, p)
        s_cur = per_probe_score(H_cur.astype(np.float32), v)
        s_test = per_probe_score(H_test.astype(np.float32), v)
        ls, auroc = find_window(s_cur, y_cur_binary, s_cur.shape[1], W)
        score_cur = np.trapz(s_cur[:, ls:ls+W], axis=1)
        score_test = np.trapz(s_test[:, ls:ls+W], axis=1)
        mu, sigma = score_cur.mean(), score_cur.std()
        pf_cur[:, pi] = (score_cur - mu) / (sigma + 1e-8)
        pf_test[:, pi] = (score_test - mu) / (sigma + 1e-8)
    sc_c = StandardScaler().fit(pf_cur)
    lr_c = LogisticRegression(C=1, max_iter=3000, class_weight="balanced").fit(
        sc_c.transform(pf_cur), y_cur_binary)
    card_scores_test = lr_c.predict_proba(sc_c.transform(pf_test))[:, 1]

    union_cur = np.concatenate([hd_feat_cur, pf_cur], axis=1)
    union_test = np.concatenate([hd_feat_test, pf_test], axis=1)
    sc_u = StandardScaler().fit(union_cur)
    lr_u = LogisticRegression(C=1, max_iter=3000, class_weight="balanced").fit(
        sc_u.transform(union_cur), y_cur_binary)
    union_scores_test = lr_u.predict_proba(sc_u.transform(union_test))[:, 1]

    R = {}
    R["HD_36D_LogReg"] = bin_metrics(y_test, hd_scores_test)
    R["CARD_8probe_LogReg"] = bin_metrics(y_test, card_scores_test)
    R["UNION_HD36_CARD8_LogReg"] = bin_metrics(y_test, union_scores_test)
    log("  T1 Overall binary AUROC:")
    for k, v in R.items():
        c = v["catch"]
        log(f"    {k:<32} AUROC={v['auroc']:.4f}  C@0.1%={c['0.001']*100:.1f}%  "
            f"C@1%={c['0.01']*100:.1f}%  C@5%={c['0.05']*100:.1f}%")
    return R

def test_t2_category(data, model_tag):
    H_cur = data["H_curated"]; y_cur = data["y_curated"]
    H_btv = data["H_btv"]; y_btv_meta = data["y_btv_meta"]; y_btv_fine = data["y_btv_fine"]
    m6 = np.isin(y_btv_meta, UNSAFE6)
    H_btv = H_btv[m6]; y_btv_meta = y_btv_meta[m6]; y_btv_fine = y_btv_fine[m6]

    unsafe_cur = np.isin(y_cur, UNSAFE6)
    H_u_cur = H_cur[unsafe_cur].astype(np.float32)
    le_u = LabelEncoder().fit(UNSAFE6)
    y_u_cur = le_u.transform(y_cur[unsafe_cur])
    ref_layers = per_cat_ref_layer(H_u_cur, y_u_cur)

    X_cur = np.concatenate([H_cur[:, ref_layers[i], :] for i in range(6)], 1).astype(np.float32)
    X_btv = np.concatenate([H_btv[:, ref_layers[i], :] for i in range(6)], 1).astype(np.float32)

    le7 = LabelEncoder().fit(["safe"] + UNSAFE6)
    y7_cur = le7.transform(y_cur)

    svd = TruncatedSVD(n_components=300, random_state=SEED).fit(X_cur)
    X_cur_lo = svd.transform(X_cur)
    X_btv_lo = svd.transform(X_btv)

    sc = StandardScaler().fit(X_cur_lo)
    Xt_cur = sc.transform(X_cur_lo)
    Xt_btv = sc.transform(X_btv_lo)

    lda = LinearDiscriminantAnalysis(n_components=6).fit(Xt_cur, y7_cur)
    lr7 = LogisticRegression(C=1, max_iter=3000, class_weight="balanced").fit(lda.transform(Xt_cur), y7_cur)
    pred_btv = lr7.predict(lda.transform(Xt_btv))
    y7_btv = le7.transform(y_btv_meta)
    acc_7way = float((pred_btv == y7_btv).mean())
    sc2 = StandardScaler().fit(X_cur_lo[unsafe_cur])
    lda6 = LinearDiscriminantAnalysis(n_components=5).fit(sc2.transform(X_cur_lo[unsafe_cur]), y_u_cur)
    svm6 = LinearSVC(C=0.1, max_iter=3000, class_weight="balanced").fit(
        lda6.transform(sc2.transform(X_cur_lo[unsafe_cur])), y_u_cur)
    pred_btv_6way = svm6.predict(lda6.transform(sc2.transform(X_btv_lo)))
    le_u = LabelEncoder().fit(UNSAFE6)
    y_btv_u = le_u.transform(y_btv_meta)
    acc_6way_std = float((pred_btv_6way == y_btv_u).mean())

    v_ref = compute_dd_probe_layers(H_cur.astype(np.float32), y_cur, "refusal")
    hd_feat_cur = per_probe_score(H_cur.astype(np.float32), v_ref)
    hd_feat_btv = per_probe_score(H_btv.astype(np.float32), v_ref)
    hd_cur_u = hd_feat_cur[unsafe_cur]
    sc_hd = StandardScaler().fit(hd_cur_u)
    lda_hd = LinearDiscriminantAnalysis(n_components=5).fit(sc_hd.transform(hd_cur_u), y_u_cur)
    svm_hd = LinearSVC(C=0.1, max_iter=3000, class_weight="balanced").fit(
        lda_hd.transform(sc_hd.transform(hd_cur_u)), y_u_cur)
    pred_hd_btv = svm_hd.predict(lda_hd.transform(sc_hd.transform(hd_feat_btv)))
    acc_hd_6way = float((pred_hd_btv == y_btv_u).mean())

    log("  T2 6-way meta-category accuracy on BTV (external test):")
    log(f"    HD  (36-D refusal) + LDA-5 + LinSVC     : {acc_hd_6way*100:.2f}%")
    log(f"    CARD (concat + SVD) + LDA-5 + LinSVC    : {acc_6way_std*100:.2f}%")
    log(f"    CARD (concat + SVD) + LDA-6 + LogReg 7way: {acc_7way*100:.2f}%")
    log(f"         (pred=safe fraction: {(pred_btv == 0).mean()*100:.1f}%)")

    return {
        "HD_6way_acc": acc_hd_6way,
        "CARD_6way_acc": acc_6way_std,
        "CARD_7way_acc_including_safe_class": acc_7way,
        "CARD_7way_pred_safe_fraction": float((pred_btv == 0).mean()),
    }

def run(model_tag):
    log(f"\n{'='*60}\n[{model_tag}] External BeaverTails-V evaluation\n{'='*60}")
    data = load_backbone_data(model_tag)
    if data is None: return None
    RES = {}
    log("\n--- T1: Binary safe/unsafe (BTV unsafe + curated safe) ---")
    RES["T1_binary"] = test_t1_binary(data, model_tag)
    log("\n--- T2: 6-way category on BTV (external, meta-mapped) ---")
    RES["T2_category"] = test_t2_category(data, model_tag)
    return RES

def main():
    ALL = {}
    for model_tag in ["qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b"]:
        res = run(model_tag)
        if res is not None:
            ALL[model_tag] = res
    if ALL:
        os.makedirs(RES_OUT, exist_ok=True)
        with open(os.path.join(RES_OUT, "beavertails_external.json"), "w") as f:
            json.dump(ALL, f, indent=2, default=str)
        log(f"\nSaved: {os.path.join(RES_OUT, 'beavertails_external.json')}")
    else:
        log("\n[WARN] No backbones had extraction files — run extract_external.py --dataset btv_eval first")

if __name__ == "__main__":
    main()
