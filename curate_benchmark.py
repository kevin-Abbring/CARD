import os, json, hashlib
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

BT_V_ROOT = os.environ.get("CARD_DATA", "data") + "/BeaverTails-V/data"
MMSB_ROOT = os.environ.get("CARD_DATA", "data") + "/MM-SafetyBench/data"
UBR_ALLAVA = os.environ.get("CARD_DATA", "data") + "/UBR_allava/dpo_test_bench.parquet"
MMVET_JSON = os.environ.get("CARD_DATA", "data") + "/MM-Vet/mm-vet.json"
MMVET_IMG = os.environ.get("CARD_DATA", "data") + "/MM-Vet/images"
OUT_DIR = os.environ.get("CARD_WORK", "outputs") + "/curated_benchmark_v2"
os.makedirs(OUT_DIR, exist_ok=True)

META_CATEGORY = {
    "violence_and_physical_harm": "violence",
    "horror_and_gore": "violence",
    "dangerous_behavior": "violence",
    "animal_abuse": "violence",
    "terrorism_or_extremism": "violence",
    "pornographic_content": "sexual",
    "sexual_crimes": "sexual",
    "insulting_and_harassing_behavior": "hate",
    "discriminatory_depictions": "hate",
    "privacy_invasion_and_surveillance": "privacy",
    "sensitive_information_in_key_areas": "privacy",
    "identity_misuse_and_impersonation": "privacy",
    "hacking_or_digital_crime": "crime",
    "financial_and_academic_fraud": "crime",
    "false_information": "misinfo",
    "deception_in_personal_relationships": "misinfo",
    "psychological_harm_and_manipulation": "misinfo",
    "Physical_Harm": "violence",
    "Sex": "sexual",
    "HateSpeech": "hate",
    "Privacy_Violence": "privacy",
    "Illegal_Activitiy": "crime",
    "Malware_Generation": "crime",
    "Fraud": "crime",
    "Political_Lobbying": "misinfo",
    "Gov_Decision": "misinfo",
}

META_CATEGORIES = ["violence", "sexual", "hate", "privacy", "crime", "misinfo"]
TARGET_PER_CATEGORY = 300
TARGET_SAFE = 500
SEED = 42

def img_bytes(v):
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, dict):
        b = v.get("bytes")
        return bytes(b) if b else None
    return None

def row_image(row, cols):
    for c in ("image", "img", "image_bytes"):
        if c in cols:
            b = img_bytes(row.get(c))
            if b is not None:
                return b
    return None

def load_beavertails_v():
    out = []
    if not os.path.isdir(BT_V_ROOT):
        return out
    for cat in sorted(os.listdir(BT_V_ROOT)):
        meta = META_CATEGORY.get(cat)
        if meta is None:
            continue
        for split in ("evaluation", "train"):
            pq = os.path.join(BT_V_ROOT, cat, f"{split}.parquet")
            if not os.path.exists(pq):
                continue
            df = pd.read_parquet(pq)
            cols = set(df.columns)
            for _, row in df.iterrows():
                if "is_response_safe" in cols and str(row.get("is_response_safe", "")).strip().lower() != "no":
                    continue
                q = str(row.get("question", "")).strip()
                if not q:
                    continue
                out.append({
                    "source": "BeaverTails-V",
                    "source_category": cat,
                    "meta_category": meta,
                    "text": q,
                    "image": row_image(row, cols),
                    "image_severity": int(row.get("image_severity", 0)) if "image_severity" in cols else 0,
                    "is_response_safe": str(row.get("is_response_safe", "")) if "is_response_safe" in cols else "",
                })
    return out

def load_mm_safetybench():
    out = []
    if not os.path.isdir(MMSB_ROOT):
        return out
    for cat in sorted(os.listdir(MMSB_ROOT)):
        meta = META_CATEGORY.get(cat)
        if meta is None:
            continue
        folder = os.path.join(MMSB_ROOT, cat)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".parquet"):
                continue
            try:
                df = pd.read_parquet(os.path.join(folder, fn))
            except Exception:
                continue
            cols = set(df.columns)
            tcol = None
            for c in ("rephrased_question", "question", "Question", "Rephrased", "prompt"):
                if c in cols:
                    tcol = c
                    break
            if tcol is None:
                continue
            for _, row in df.iterrows():
                q = str(row.get(tcol, "")).strip()
                if not q:
                    continue
                out.append({
                    "source": "MM-SafetyBench",
                    "source_category": cat,
                    "meta_category": meta,
                    "text": q,
                    "image": row_image(row, cols),
                    "image_severity": 0,
                    "is_response_safe": "",
                })
    return out

def load_allava_safe():
    out = []
    if not os.path.exists(UBR_ALLAVA):
        return out
    df = pd.read_parquet(UBR_ALLAVA)
    cols = set(df.columns)
    for _, row in df.iterrows():
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        out.append({
            "source": "UBR_allava",
            "source_category": str(row.get("task_type", "unknown")),
            "meta_category": "safe",
            "text": prompt,
            "image": row_image(row, cols),
            "image_severity": 0,
            "is_response_safe": "yes",
        })
    return out

def load_mmvet_safe():
    out = []
    if not os.path.exists(MMVET_JSON):
        return out
    with open(MMVET_JSON) as f:
        data = json.load(f)
    for key, item in data.items():
        prompt = str(item.get("question", "")).strip()
        if not prompt:
            continue
        image = None
        name = item.get("imagename")
        if name:
            p = os.path.join(MMVET_IMG, name)
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    image = fh.read()
        out.append({
            "source": "MM-Vet",
            "source_category": "benign_qa",
            "meta_category": "safe",
            "text": prompt,
            "image": image,
            "image_severity": 0,
            "is_response_safe": "yes",
        })
    return out

def sample_uniform(pool, k, rng):
    pool = sorted(pool, key=lambda r: hashlib.md5((r["source"] + "|" + r["source_category"] + "|" + r["text"]).encode("utf-8")).hexdigest())
    if len(pool) <= k:
        return pool
    idx = sorted(rng.choice(len(pool), size=k, replace=False).tolist())
    return [pool[i] for i in idx]

def main():
    rng = np.random.default_rng(SEED)
    candidates = load_beavertails_v() + load_mm_safetybench() + load_allava_safe() + load_mmvet_safe()
    by_meta = defaultdict(list)
    for c in candidates:
        by_meta[c["meta_category"]].append(c)
    print(f"candidate pool: {len(candidates)}", flush=True)
    for mc in sorted(by_meta):
        n_img = sum(1 for c in by_meta[mc] if c["image"] is not None)
        srcs = dict(Counter(c["source"] for c in by_meta[mc]))
        print(f"  {mc:<10} pool={len(by_meta[mc]):>6}  with_image={n_img:>6}  sources={srcs}", flush=True)

    curated = []
    for mc in META_CATEGORIES:
        picked = sample_uniform(by_meta.get(mc, []), TARGET_PER_CATEGORY, rng)
        curated.extend(picked)
        n_img = sum(1 for c in picked if c["image"] is not None)
        print(f"sample {mc:<10} n={len(picked):>4}  with_image={n_img:>4}", flush=True)
    safe = sample_uniform(by_meta.get("safe", []), TARGET_SAFE, rng)
    curated.extend(safe)
    n_img = sum(1 for c in safe if c["image"] is not None)
    print(f"sample {'safe':<10} n={len(safe):>4}  with_image={n_img:>4}", flush=True)

    manifest = pd.DataFrame(curated, columns=["source", "source_category", "meta_category", "text", "image", "image_severity", "is_response_safe"])
    out_path = os.path.join(OUT_DIR, "curated_manifest.parquet")
    manifest.to_parquet(out_path, index=False)
    print(f"saved {out_path}  total={len(manifest)}", flush=True)
    print(manifest["meta_category"].value_counts().to_string(), flush=True)
    cov = manifest["image"].notna().mean() if len(manifest) else 0.0
    print(f"overall image coverage: {cov*100:.1f}%", flush=True)

if __name__ == "__main__":
    main()
