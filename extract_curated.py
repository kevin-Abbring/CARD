import os, argparse, pickle, io
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from tqdm import tqdm

OUT_DIR = os.environ.get("CARD_WORK", "outputs") + "/curated_benchmark_v2"
MANIFEST = os.path.join(OUT_DIR, "curated_manifest.parquet")

NPY_NAME = {
    "qwen3vl_2b": "hidden_states_qwen3vl_2b_curated_v2.npy",
    "qwen3vl_4b": "hidden_states_qwen3vl_4b_curated_v2.npy",
    "qwen3vl_8b": "hidden_states_qwen3vl_8b_curated_v2.npy",
    "gemma3_12b": "hidden_states_gemma3_12b_curated_v2.npy",
    "llava15_7b": "hidden_states_llava15_7b_curated_v2.npy",
}
META_NAME = {
    "qwen3vl_2b": "hidden_states_qwen3vl_2b_meta.pkl",
    "qwen3vl_4b": "hidden_states_qwen3vl_4b_meta.pkl",
    "qwen3vl_8b": "hidden_states_qwen3vl_8b_meta.pkl",
    "gemma3_12b": "hidden_states_gemma3_12b_meta.pkl",
    "llava15_7b": "hidden_states_llava15_7b_meta.pkl",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLAVA_SYS = ("A chat between a curious human and an artificial intelligence assistant. "
             "The assistant gives helpful, detailed, and polite answers to the human's questions.")

def to_pil(b):
    if b is None:
        return None
    try:
        return PILImage.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return None

def load_manifest():
    df = pd.read_parquet(MANIFEST)
    print(f"manifest: {len(df)} samples; cols={list(df.columns)}", flush=True)
    return df

def meta_row(row, has_image):
    return {
        "text": row["text"],
        "meta_category": row["meta_category"],
        "source": row.get("source", "unknown"),
        "source_category": row.get("source_category", "unknown"),
        "has_image": bool(has_image),
    }

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
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_manifest()
    has_image_col = "image" in df.columns
    model, processor = load_model(args.family, args.model_path)
    N = len(df)
    H = None
    meta_out = []
    err = 0
    n_img = 0
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=N, desc=f"fwd[{args.family}]")):
        text = str(row["text"])
        image = to_pil(row["image"]) if has_image_col else None
        try:
            inputs = build_inputs(args.family, processor, text, image).to(DEVICE)
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
            meta_out.append(meta_row(row, image is not None))
            n_img += int(image is not None)
            del out, inputs
        except Exception as e:
            err += 1
            meta_out.append({**meta_row(row, image is not None), "ERROR": str(e)[:200]})
            if err <= 3:
                print(f"[err {i}] {e}", flush=True)
        if image is not None:
            image.close()
        if (i + 1) % 100 == 0 and DEVICE == "cuda":
            torch.cuda.empty_cache()
    npy = os.path.join(OUT_DIR, NPY_NAME[args.tag])
    mpk = os.path.join(OUT_DIR, META_NAME[args.tag])
    np.save(npy, H)
    with open(mpk, "wb") as f:
        pickle.dump({"meta": meta_out, "num_layers": int(H.shape[1]), "hidden_size": int(H.shape[2])}, f)
    print(f"saved {npy} shape={H.shape} images_seen={n_img}/{N} errs={err}", flush=True)

if __name__ == "__main__":
    main()
