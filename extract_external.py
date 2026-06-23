import os, argparse, pickle, io
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from tqdm import tqdm

OUT_DIR = os.environ.get("CARD_WORK", "outputs") + "/external_eval"
BTV_ROOT = os.environ.get("CARD_DATA", "data") + "/BeaverTails-V/data"
MMSB_ROOT = os.environ.get("CARD_DATA", "data") + "/MM-SafetyBench/data"
JBV_ROOT = os.environ.get("CARD_DATA", "data") + "/JailBreakV_28K"
JBV_CSV = os.environ.get("CARD_DATA", "data") + "/JailBreakV_28K/mini_JailBreakV_28K.csv"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLAVA_SYS = ("A chat between a curious human and an artificial intelligence assistant. "
             "The assistant gives helpful, detailed, and polite answers to the human's questions.")

def img_bytes(v):
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, dict):
        b = v.get("bytes")
        return bytes(b) if b else None
    return None

def to_pil(b):
    if b is None:
        return None
    try:
        return PILImage.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return None

def load_btv_eval(root, max_per_cat):
    rows = []
    if not os.path.isdir(root):
        return rows
    for cat in sorted(os.listdir(root)):
        p = os.path.join(root, cat, "evaluation.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        if max_per_cat and len(df) > max_per_cat:
            df = df.sample(n=max_per_cat, random_state=SEED)
        for _, r in df.iterrows():
            q = str(r.get("question", "")).strip()
            if not q:
                continue
            rows.append({"question": q, "image": img_bytes(r.get("image")), "category": cat,
                         "attack_type": "btv", "is_response_safe": str(r.get("is_response_safe", ""))})
    return rows

def load_mm_safety(root, max_per_cat_attack):
    rows = []
    if not os.path.isdir(root):
        return rows
    for cat in sorted(os.listdir(root)):
        cp = os.path.join(root, cat)
        if not os.path.isdir(cp):
            continue
        for attack in ["Text_only", "SD", "TYPO", "SD_TYPO"]:
            p = os.path.join(cp, f"{attack}.parquet")
            if not os.path.exists(p):
                continue
            df = pd.read_parquet(p)
            tcol = next((c for c in ("rephrased_question", "question", "Question", "Rephrased", "prompt") if c in df.columns), None)
            if tcol is None:
                continue
            if max_per_cat_attack and len(df) > max_per_cat_attack:
                df = df.sample(n=max_per_cat_attack, random_state=SEED)
            for _, r in df.iterrows():
                q = str(r.get(tcol, "")).strip()
                if not q:
                    continue
                rows.append({"question": q, "image": img_bytes(r.get("image")), "category": cat,
                             "attack_type": attack, "is_response_safe": ""})
    return rows

def load_jbv(csv, root, max_n):
    rows = []
    if not os.path.exists(csv):
        return rows
    df = pd.read_csv(csv)
    if max_n and len(df) > max_n:
        df = df.sample(n=max_n, random_state=SEED)
    for _, r in df.iterrows():
        q = str(r.get("jailbreak_query", "")).strip()
        if not q:
            continue
        ib = None
        rel = str(r.get("image_path", ""))
        if rel and not rel.startswith(("None", "nan")):
            ip = os.path.join(root, rel)
            if os.path.isfile(ip):
                try:
                    with open(ip, "rb") as f:
                        ib = f.read()
                except Exception:
                    ib = None
        rows.append({"question": q, "image": ib, "category": str(r.get("policy", "unknown")),
                     "attack_type": str(r.get("format", "unknown")), "is_response_safe": ""})
    return rows

def load_model(family, model_path):
    from transformers import AutoProcessor
    if family == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration as M
    elif family == "gemma3":
        from transformers import Gemma3ForConditionalGeneration as M
    elif family == "llava":
        from transformers import LlavaForConditionalGeneration as M
    else:
        raise ValueError(family)
    model = M.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).eval().to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if family == "llava":
        try:
            processor.patch_size = model.config.vision_config.patch_size
            processor.vision_feature_select_strategy = getattr(model.config, "vision_feature_select_strategy", "default")
            processor.num_additional_image_tokens = 1
        except Exception as e:
            print(f"[warn] llava processor cfg: {e}", flush=True)
    return model, processor

def build_inputs(family, processor, text, image):
    if family == "qwen3vl":
        if image is not None:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return processor(text=[chat], images=[image], return_tensors="pt")
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[chat], return_tensors="pt")
    if family == "gemma3":
        if image is not None:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    if family == "llava":
        if image is not None:
            prompt = f"{LLAVA_SYS} USER: <image>\n{text} ASSISTANT:"
            return processor(images=image, text=prompt, return_tensors="pt")
        prompt = f"{LLAVA_SYS} USER: {text} ASSISTANT:"
        return processor(text=prompt, return_tensors="pt")
    raise ValueError(family)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["qwen3vl", "gemma3", "llava"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", required=True, choices=["btv_eval", "mm_safety", "jailbreakv"])
    ap.add_argument("--max_per_cat", type=int, default=0)
    ap.add_argument("--max_per_cat_attack", type=int, default=30)
    ap.add_argument("--max_jbv", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.dataset == "btv_eval":
        rows = load_btv_eval(BTV_ROOT, args.max_per_cat)
    elif args.dataset == "mm_safety":
        rows = load_mm_safety(MMSB_ROOT, args.max_per_cat_attack)
    else:
        rows = load_jbv(JBV_CSV, JBV_ROOT, args.max_jbv)
    print(f"{args.dataset}: {len(rows)} samples", flush=True)

    model, processor = load_model(args.family, args.model_path)
    N = len(rows)
    H = None
    meta_out = []
    err = 0
    n_img = 0
    for i, r in enumerate(tqdm(rows, total=N, desc=f"{args.dataset}[{args.family}]")):
        image = to_pil(r["image"])
        try:
            inputs = build_inputs(args.family, processor, r["question"], image).to(DEVICE)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            nl = len(hs) - 1
            if H is None:
                hidden = hs[-1].shape[-1]
                H = np.zeros((N, nl, hidden), dtype=np.float16)
                print(f"num_layers={nl} hidden={hidden}", flush=True)
            for l in range(nl):
                H[i, l] = hs[l + 1][0, -1, :].float().cpu().numpy().astype(np.float16)
            meta_out.append({"question": r["question"][:200], "category": r["category"],
                             "attack_type": r["attack_type"], "is_response_safe": r.get("is_response_safe", ""),
                             "has_image": image is not None})
            n_img += int(image is not None)
            del out, inputs
        except Exception as e:
            err += 1
            meta_out.append({"question": r["question"][:200], "category": r["category"],
                             "attack_type": r["attack_type"], "ERROR": str(e)[:200]})
            if err <= 3:
                print(f"[err {i}] {e}", flush=True)
        if image is not None:
            image.close()
        if (i + 1) % 50 == 0 and DEVICE == "cuda":
            torch.cuda.empty_cache()
    npy = os.path.join(OUT_DIR, f"{args.dataset}_hidden_states_{args.tag}.npy")
    mpk = os.path.join(OUT_DIR, f"{args.dataset}_meta_{args.tag}.pkl")
    np.save(npy, H)
    with open(mpk, "wb") as f:
        pickle.dump({"meta": meta_out, "num_layers": int(H.shape[1]), "hidden_size": int(H.shape[2])}, f)
    print(f"saved {npy} shape={H.shape} images_seen={n_img}/{N} errs={err}", flush=True)

if __name__ == "__main__":
    main()
