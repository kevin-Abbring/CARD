# CARD: Category-Aware Risk Detection for Vision-Language Models

Reference implementation of the hidden-state probing pipeline in *CARD:
Category-Aware Risk Detection for Vision-Language Models via Depth-Localized
Hidden-State Probing*.

CARD reads frozen VLM hidden states from a **multimodal (image + text) prefill**
and trains lightweight probes to (i) flag unsafe inputs and (ii) predict their
harm category, across five backbones from three families
(Qwen3-VL-2B/4B/8B, Gemma-3-12B, LLaVA-1.5-7B). The VLM backbone stays frozen;
only the small category head is trained over cached hidden states.

## Run

```bash
bash run_curate.sh     # build the test-set manifest
bash run_extract.sh    # multimodal extraction + in-domain analysis
bash run_external.sh   # external-shift extraction + analysis
```

Set `MODEL_DIR` and `GPU` to your local model directory and device. Dataset and
output locations are configured at the top of each script. External-shift
evaluation (`run_external.sh`) is run for the Qwen3-VL backbones (2B/4B/8B).

## Requirements

See `requirements.txt` (PyTorch, Transformers ≥ 4.57, scikit-learn, etc.).
