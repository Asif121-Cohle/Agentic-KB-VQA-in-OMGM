# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:28:06.893438Z","iopub.execute_input":"2026-10-22T06:28:06.893713Z","iopub.status.idle":"2026-10-22T06:28:14.895544Z","shell.execute_reply.started":"2026-10-22T06:28:06.893690Z","shell.execute_reply":"2026-10-22T06:28:14.894539Z"}}
!pip install -q -U bitsandbytes accelerate peft


# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:28:14.897376Z","iopub.execute_input":"2026-10-22T06:28:14.897685Z","iopub.status.idle":"2026-10-22T06:28:36.892564Z","shell.execute_reply.started":"2026-10-22T06:28:14.897660Z","shell.execute_reply":"2026-10-22T06:28:36.891879Z"}}
!pip install -q sentencepiece protobuf
!pip install -q timm einops fairscale faiss-cpu iopath decord webdataset omegaconf pycocoevalcap


# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:31:31.802394Z","iopub.execute_input":"2026-10-22T06:31:31.802977Z","iopub.status.idle":"2026-10-22T06:32:02.549106Z","shell.execute_reply.started":"2026-10-22T06:31:31.802947Z","shell.execute_reply":"2026-10-22T06:32:02.548480Z"}}

import subprocess, sys
def ensure_package(pkg, imp=None):
    try:
        __import__(imp or pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", pkg])

ensure_package("bitsandbytes")
ensure_package("accelerate")
ensure_package("peft")
print("All dependencies ready.")


# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:33:38.296712Z","iopub.execute_input":"2026-10-22T06:33:38.297287Z","iopub.status.idle":"2026-10-22T06:33:38.819610Z","shell.execute_reply.started":"2026-10-22T06:33:38.297260Z","shell.execute_reply":"2026-10-22T06:33:38.818761Z"}}

# %% [markdown]
# ### Cell 1: Imports, Seeds & Path Configuration

# %%
import os
import gc
import json
import re
import string
import time
import glob
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import PIL.Image
import torch
from tqdm import tqdm

print("=" * 70)
print("CELL 1: IMPORTS, SEEDS & PATH CONFIGURATION")
print("=" * 70)

# --- Deterministic Reproducibility ---
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Seed set to {SEED} across all libraries.")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU [{i}]: {props.name} | VRAM: {props.total_memory / 1024**3:.2f} GB")

# --- Path Configuration (from kaggle_ds_paths.md) ---
BASE_DS_DIR = "/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds"
ALT_DS_DIR  = "/kaggle/input/agentic-vqa-ds"
LOCAL_PRE   = "pre helper"

def resolve_path(candidates: List[str], is_dir: bool = False) -> str:
    for p in candidates:
        if os.path.exists(p):
            if is_dir and os.path.isdir(p): return p
            elif not is_dir and os.path.isfile(p): return p
            elif os.path.exists(p): return p
    return candidates[0]

STEP2_PATH    = resolve_path([f"{BASE_DS_DIR}/result_step_2.json", f"{BASE_DS_DIR}/result_step2.json", f"{ALT_DS_DIR}/result_step_2.json", f"{LOCAL_PRE}/res/result_step2.json"])
STEP3_PATH    = resolve_path([f"{BASE_DS_DIR}/step3_predictions.json", f"{ALT_DS_DIR}/step3_predictions.json", f"{LOCAL_PRE}/res/step3_predictions.json"])
SAMPLE_PATH   = resolve_path([f"{BASE_DS_DIR}/infoseek_test_sample_1000.csv", f"{ALT_DS_DIR}/infoseek_test_sample_1000.csv", f"{LOCAL_PRE}/files/infoseek_test_sample_1000.csv"])
KB_PATH       = resolve_path([f"{BASE_DS_DIR}/infoseek_100k_wiki/wiki_100_dict_v4.json", f"{ALT_DS_DIR}/infoseek_100k_wiki/wiki_100_dict_v4.json"])
QUERY_IMG_DIR = resolve_path([f"{BASE_DS_DIR}/infoseek_val_1000_images/infoseek_val_1000_images", f"{BASE_DS_DIR}/infoseek_val_1000_images", f"{ALT_DS_DIR}/infoseek_val_1000_images/infoseek_val_1000_images"], is_dir=True)
LORA_ADAPTER  = resolve_path([f"{BASE_DS_DIR}/llava_verifier_lora", f"{ALT_DS_DIR}/llava_verifier_lora", "/kaggle/working/llava_1_5_7b_agentic_verifier_lora"], is_dir=True)
OUTPUT_DIR    = "/kaggle/working/agentic_results_finetuned"
TAU           = 0.025

print(f"\nResolved Paths:")
print(f"  Step 2 Result : {STEP2_PATH}  (exists: {os.path.exists(STEP2_PATH)})")
print(f"  Step 3 Baseline: {STEP3_PATH}  (exists: {os.path.exists(STEP3_PATH)})")
print(f"  Sample CSV    : {SAMPLE_PATH}  (exists: {os.path.exists(SAMPLE_PATH)})")
print(f"  KB JSON       : {KB_PATH}  (exists: {os.path.exists(KB_PATH)})")
print(f"  Query Images  : {QUERY_IMG_DIR}  (exists: {os.path.isdir(QUERY_IMG_DIR)})")
print(f"  LoRA Adapter  : {LORA_ADAPTER}  (exists: {os.path.isdir(LORA_ADAPTER)})")
print(f"  Tau Threshold : {TAU}")
os.makedirs(OUTPUT_DIR, exist_ok=True)



# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:33:43.954786Z","iopub.execute_input":"2026-10-22T06:33:43.955415Z","iopub.status.idle":"2026-10-22T06:33:43.965234Z","shell.execute_reply.started":"2026-10-22T06:33:43.955386Z","shell.execute_reply":"2026-10-22T06:33:43.964340Z"}}
# %% [markdown]
# ### Cell 2: Evaluation Metric Helpers (OMGM official `check_match`)

# %%
print("\n" + "=" * 70)
print("CELL 2: INFOSEEK EVALUATION METRIC HELPERS")
print("=" * 70)

def normalize_answer(text: Any) -> str:
    if not text: return ""
    text = str(text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.lower().split())

def extract_number(text: Any) -> Optional[float]:
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(text).replace(",", ""))
    try: return float(matches[0]) if matches else None
    except ValueError: return None

def metric_numerical_range(pred: float, ans: float, tol: float = 0.10) -> bool:
    lo = ans * (1 - tol) if ans >= 0 else ans * (1 + tol)
    hi = ans * (1 + tol) if ans >= 0 else ans * (1 - tol)
    return min(lo, hi) <= pred <= max(lo, hi)

def check_match(pred: str, gt: Any) -> bool:
    if not pred or not gt: return False
    np_ = normalize_answer(pred)
    if not np_: return False
    gt_list = gt if isinstance(gt, list) else [g.strip() for g in str(gt).split("|") if str(g).strip()]
    for g in gt_list:
        ng = normalize_answer(g)
        if ng and (ng == np_ or f" {ng} " in f" {np_} " or ng in np_): return True
    pn = extract_number(pred)
    if pn is not None:
        for g in gt_list:
            c = str(g).strip().replace(",", "")
            if re.fullmatch(r"[-+]?\d*\.?\d+", c):
                if metric_numerical_range(pn, float(c)): return True
    return False

print("Evaluation helpers loaded (normalize_answer, check_match with ±10% numeric tolerance).")



# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:33:51.433953Z","iopub.execute_input":"2026-10-22T06:33:51.434474Z","iopub.status.idle":"2026-10-22T06:33:51.443744Z","shell.execute_reply.started":"2026-10-22T06:33:51.434444Z","shell.execute_reply":"2026-10-22T06:33:51.442975Z"}}
# %% [markdown]
# ### Cell 3: Knowledge Base & Query Image Helpers

# %%
print("\n" + "=" * 70)
print("CELL 3: KNOWLEDGE BASE & QUERY IMAGE HELPERS")
print("=" * 70)

class KnowledgeEntry:
    def __init__(self, title, url, section_titles, section_texts):
        self.title = title; self.url = url
        self.section_titles = section_titles; self.section_texts = section_texts

def reconstruct_wiki_sections(entry: KnowledgeEntry) -> List[str]:
    sections = []
    for i, sec_t in enumerate(entry.section_titles):
        sl = sec_t.lower()
        if "external links" in sl or "references" in sl: continue
        text = entry.section_texts[i] if i < len(entry.section_texts) else ""
        sections.append(f"# Wiki Article: {entry.title}\n## Section Title: {sec_t}\n{text}")
    return sections

def get_query_image(image_id: str, search_dir: str) -> PIL.Image.Image:
    clean = str(image_id).split("|")[0].strip()
    for ext in ["", ".jpg", ".jpeg", ".JPEG", ".png"]:
        p = os.path.join(search_dir, f"{clean}{ext}")
        if os.path.isfile(p):
            try: return PIL.Image.open(p).convert("RGB")
            except Exception: pass
    for m in glob.glob(os.path.join(search_dir, f"*{clean}*")):
        if os.path.isfile(m) and m.lower().endswith((".jpg", ".jpeg", ".png")):
            try: return PIL.Image.open(m).convert("RGB")
            except Exception: pass
    return PIL.Image.new("RGB", (224, 224), color=(128, 128, 128))

print("KnowledgeEntry class and image helpers loaded.")



# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:33:56.505706Z","iopub.execute_input":"2026-10-22T06:33:56.506253Z","iopub.status.idle":"2026-10-22T06:34:14.927117Z","shell.execute_reply.started":"2026-10-22T06:33:56.506209Z","shell.execute_reply":"2026-10-22T06:34:14.926261Z"}}
# %% [markdown]
# ### Cell 4: BGE Section Reranker Initialization

# %%
print("\n" + "=" * 70)
print("CELL 4: BGE SECTION RERANKER")
print("=" * 70)

bge_model = None
bge_tokenizer = None

def init_bge_reranker(device="cuda"):
    global bge_model, bge_tokenizer
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        bge_id = "BAAI/bge-reranker-v2-m3"
        print(f"Loading BGE Reranker ({bge_id}) on {device}...")
        bge_tokenizer = AutoTokenizer.from_pretrained(bge_id)
        bge_model = AutoModelForSequenceClassification.from_pretrained(bge_id).half().to(device).eval()
        print("BGE Reranker loaded (FP16, ~560 MB VRAM).")
    except Exception as e:
        print(f"BGE not loaded ({e}). Lexical fallback will be used.")

def select_best_section(question: str, entry: Optional[KnowledgeEntry]) -> Tuple[str, float]:
    if not entry: return "", -999.0
    sections = reconstruct_wiki_sections(entry)
    if not sections: return "", -999.0
    if len(sections) == 1: return sections[0], 0.0
    if bge_model is not None and torch.cuda.is_available():
        try:
            pairs = [[question, s[:1500]] for s in sections]
            from transformers import AutoTokenizer
            inp = bge_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
            with torch.no_grad():
                scores = bge_model(**inp).logits.view(-1).float().cpu().tolist()
            best = int(np.argmax(scores))
            return sections[best], float(scores[best])
        except Exception: pass
    # Lexical fallback
    stopwords = {"what","is","the","of","this","in","how","many","a","an","to","for","does","do","which","where","who","when","why"}
    qw = [w.lower() for w in re.findall(r"\b\w+\b", question) if w.lower() not in stopwords]
    best_sec, best_sc = sections[0], -1.0
    for sec in sections:
        sl = sec.lower()
        sc = sum(5.0 if w in sl.split("\n")[1] else 1.0 if w in sl else 0.0 for w in qw)
        if sc > best_sc: best_sc, best_sec = sc, sec
    return best_sec, float(best_sc)

init_bge_reranker("cuda" if torch.cuda.is_available() else "cpu")



# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:34:43.217171Z","iopub.execute_input":"2026-10-22T06:34:43.217650Z","iopub.status.idle":"2026-10-22T06:36:34.773651Z","shell.execute_reply.started":"2026-10-22T06:34:43.217623Z","shell.execute_reply":"2026-10-22T06:36:34.772695Z"}}
# %% [markdown]
# ### Cell 5: LLaVA-1.5-7B Base Model + LoRA Adapter Loading
# Loads the 4-bit base model first, then attaches the fine-tuned LoRA adapter.
# Both answer generation and visual verification share the same base model.
# The LoRA adapter adds **zero** extra VRAM — it's only ~161 MB of weight deltas.

# %%
print("\n" + "=" * 70)
print("CELL 5: LLaVA-1.5-7B BASE MODEL + LoRA ADAPTER LOADING")
print("=" * 70)

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"1. Loading processor from '{MODEL_ID}'...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.unk_token
processor.tokenizer.padding_side = "right"

print("2. Loading base LLaVA-1.5-7B in 4-bit NF4...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
base_model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
).eval()

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  After base model — GPU [{i}]: {alloc:.2f} GB allocated")

print(f"\n3. Attaching Fine-Tuned LoRA Adapter from: '{LORA_ADAPTER}'")
assert os.path.isdir(LORA_ADAPTER), f"LoRA adapter directory not found: {LORA_ADAPTER}"
verifier_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
verifier_model.eval()
print("   LoRA adapter attached successfully.")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  After LoRA attach — GPU [{i}]: {alloc:.2f} GB allocated")

print("\n4. Verification: Confirm adapter config keys:")
print(f"   Adapter trainable params visible: {sum(p.numel() for p in verifier_model.parameters() if p.requires_grad):,}")



# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:40:17.342637Z","iopub.execute_input":"2026-10-22T06:40:17.343333Z","iopub.status.idle":"2026-10-22T06:40:17.370300Z","shell.execute_reply.started":"2026-10-22T06:40:17.343304Z","shell.execute_reply":"2026-10-22T06:40:17.369521Z"}}
print("\n4. Verification: Confirm LoRA Adapter is Active (Inference Mode)")
# In .eval() mode all params have requires_grad=False — this is correct and expected.
# The real check: total LoRA weight count and active adapter list.
total_lora_params = sum(p.numel() for p in verifier_model.parameters())
lora_delta_params = total_lora_params - sum(p.numel() for p in base_model.parameters())
print(f"   Total model params (base + LoRA deltas): {total_lora_params:,}")
print(f"   Estimated LoRA delta params: {lora_delta_params:,}")
print(f"   Active adapter(s): {list(verifier_model.peft_config.keys())}")
print(f"   Adapter type: {list(verifier_model.peft_config.values())[0].peft_type}")
# Confirm safetensors file is present in the adapter directory
adapter_weights_file = os.path.join(LORA_ADAPTER, "adapter_model.safetensors")
print(f"   Adapter weights file: {adapter_weights_file} (exists: {os.path.exists(adapter_weights_file)})")
print("   NOTE: requires_grad=False is CORRECT for inference — LoRA weights are active and applied.")
assert os.path.exists(adapter_weights_file), "CRITICAL: adapter_model.safetensors missing!"


# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:41:06.168873Z","iopub.execute_input":"2026-10-22T06:41:06.169418Z","iopub.status.idle":"2026-10-22T06:41:18.674954Z","shell.execute_reply.started":"2026-10-22T06:41:06.169390Z","shell.execute_reply":"2026-10-22T06:41:18.673938Z"}}
print("\n" + "=" * 70)
print("CELL 5: LLaVA-1.5-7B BASE MODEL + LoRA ADAPTER LOADING")
print("=" * 70)

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"1. Loading processor from '{MODEL_ID}'...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.unk_token
processor.tokenizer.padding_side = "right"

print("2. Loading base LLaVA-1.5-7B in 4-bit NF4...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
base_model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
).eval()

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  After base model — GPU [{i}]: {alloc:.2f} GB allocated")

print(f"\n3. Attaching Fine-Tuned LoRA Adapter from: '{LORA_ADAPTER}'")
assert os.path.isdir(LORA_ADAPTER), f"LoRA adapter directory not found: {LORA_ADAPTER}"
verifier_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
verifier_model.eval()
print("   LoRA adapter attached successfully.")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  After LoRA attach — GPU [{i}]: {alloc:.2f} GB allocated")

print("\n4. Verification: Confirm LoRA Adapter is Active (Inference Mode)")
# In .eval() mode all params have requires_grad=False — this is correct and expected.
# The real check: total LoRA weight count and active adapter list.
total_lora_params = sum(p.numel() for p in verifier_model.parameters())
lora_delta_params = total_lora_params - sum(p.numel() for p in base_model.parameters())
print(f"   Total model params (base + LoRA deltas): {total_lora_params:,}")
print(f"   Estimated LoRA delta params: {lora_delta_params:,}")
print(f"   Active adapter(s): {list(verifier_model.peft_config.keys())}")
print(f"   Adapter type: {list(verifier_model.peft_config.values())[0].peft_type}")
# Confirm safetensors file is present in the adapter directory
adapter_weights_file = os.path.join(LORA_ADAPTER, "adapter_model.safetensors")
print(f"   Adapter weights file: {adapter_weights_file} (exists: {os.path.exists(adapter_weights_file)})")
print("   NOTE: requires_grad=False is CORRECT for inference — LoRA weights are active and applied.")
assert os.path.exists(adapter_weights_file), "CRITICAL: adapter_model.safetensors missing!"


# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:43:42.029445Z","iopub.execute_input":"2026-10-22T06:43:42.030291Z","iopub.status.idle":"2026-10-22T06:43:42.041168Z","shell.execute_reply.started":"2026-10-22T06:43:42.030263Z","shell.execute_reply":"2026-10-22T06:43:42.040370Z"}}
# %% [markdown]
# ### Cell 6: Inference Helper Functions
# `ask_visual_choice_finetuned` — replaces the two-pass zero-shot debiaser.
# Because the adapter was trained with 50/50 symmetric pairs, a SINGLE-PASS
# prompt is now sufficient and unbiased. Two passes are no longer needed.

# %%
print("\n" + "=" * 70)
print("CELL 6: INFERENCE HELPER FUNCTIONS")
print("=" * 70)

OMGM_INSTRUCTION = (
    "Answer the encyclopedic question about the given image.\n"
    "Don't mention the visual content of image in your output. Directly output the answer of the question according to the Context.\n"
    "If you need to answer questions about numbers or time, please output the corresponding numerical format directly.\n"
    "Pay special attention to units: if the question asks for metres and context mentions centimetres, convert to metres (e.g., 125 cm is 1.25 metres).\n"
    "If the context does not contain the information required to answer the question, you should answer the question using internal model knowledge.\n"
    "There is an example:\n- Context: # Wiki Article: Dolomites\n## Section Title: Dolomites\nThe Dolomites, also known as the Dolomite Mountains, Dolomite Alps or Dolomitic Alps, are a mountain range located in northeastern Italy. The Dolomites are located in the regions of Veneto, Trentino-Alto Adige/Südtirol and Friuli Venezia Giulia, covering an area shared between the provinces of Belluno, Vicenza, Verona, Trentino, South Tyrol, Udine and Pordenone.\n"
    "- Question: Which city or region does this mountain locate in?\n"
    "Just answer the questions , no explanations needed. Short answer is: Province of Belluno\n\n"
)

def generate_answer(img: PIL.Image.Image, question: str, section_text: str, max_chars: int = 2500) -> str:
    """Generate factual answer using the BASE model (no LoRA adapter — standard OMGM generation)."""
    trimmed = section_text[:max_chars]
    prompt_text = OMGM_INSTRUCTION + f"- Context: {trimmed}\n- Question: {question}\nJust answer the questions , no explanations needed. Short answer is:"
    full_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
    inputs = processor(images=img, text=full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        # Use base_model (without LoRA) for generation to maintain answer style parity
        out = base_model.generate(**inputs, max_new_tokens=32, do_sample=False)
    plen = inputs.input_ids.shape[1]
    return processor.decode(out[0][plen:], skip_special_tokens=True).strip()

def ask_visual_choice_finetuned(img: PIL.Image.Image, title1: str, title2: str) -> Tuple[str, str]:
    """
    Single-pass fine-tuned verifier (replaces two-pass zero-shot debiaser).
    The LoRA adapter (trained on 50/50 symmetric pairs) has primacy bias < 1.5%,
    so a single forward pass is accurate and unbiased.
    Returns: ("A" -> keep Candidate 1, "B" -> switch to Candidate 2), debug_string
    """
    prompt_text = (
        f"<image>\n"
        f"Look at the image carefully.\n"
        f"Which entity does this image depict?\n"
        f"Option (A): {title1}\n"
        f"Option (B): {title2}\n"
        f"Answer strictly with (A) or (B):"
    )
    full_prompt = f"USER: {prompt_text}\nASSISTANT:"
    inputs = processor(images=img, text=full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        # Use verifier_model (base + LoRA adapter) for the entity choice
        out = verifier_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    plen = inputs.input_ids.shape[1]
    generated = processor.decode(out[0][plen:], skip_special_tokens=True).strip()
    gen_lower = generated.lower()

    if gen_lower.startswith("(b)") or gen_lower.startswith("b)") or gen_lower.startswith("b ") or gen_lower == "b":
        return "B", f"FineTuned→Cand2 | gen: {generated}"
    return "A", f"FineTuned→Cand1 | gen: {generated}"

print("Inference helpers loaded:")
print("  - generate_answer()            → base_model (no LoRA, OMGM-style VQA)")
print("  - ask_visual_choice_finetuned() → verifier_model (base + LoRA, single-pass)")



# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:43:42.042736Z","iopub.execute_input":"2026-10-22T06:43:42.043105Z","iopub.status.idle":"2026-10-22T06:43:43.415551Z","shell.execute_reply.started":"2026-10-22T06:43:42.043072Z","shell.execute_reply":"2026-10-22T06:43:43.414850Z"}}
# %% [markdown]
# ### Cell 7: Data Loading & Baseline Analysis

# %%
print("\n" + "=" * 70)
print("CELL 7: DATA LOADING & BASELINE ANALYSIS")
print("=" * 70)

with open(STEP2_PATH, "r") as f: step2_results = json.load(f)
with open(STEP3_PATH, "r") as f: step3_baseline = json.load(f)
df_samples = pd.read_csv(SAMPLE_PATH)
sample_meta = {
    str(row.get("data_id", "")).strip(): {
        "question": str(row.get("question", "")),
        "image_id": str(row.get("dataset_image_ids", "")).split("|")[0].strip(),
        "ground_truth_answers": str(row.get("answer", "")),
    }
    for _, row in df_samples.iterrows()
}
print(f"Loaded: {len(step2_results):,} Step 2 entries | {len(step3_baseline):,} Step 3 entries | {len(sample_meta):,} sample metadata rows")

# Compute score margins and identify ambiguous queries
margins, ambiguous_qids = [], []
baseline_correct = 0
for did, pinfo in step3_baseline.items():
    gt = pinfo.get("ground_truth_answers") or sample_meta.get(did, {}).get("ground_truth_answers", "")
    if check_match(pinfo.get("prediction", ""), gt): baseline_correct += 1
    s2 = step2_results.get(did, {})
    scores = s2.get("retrieval_similarities", [])
    delta = float(scores[0] - scores[1]) if len(scores) > 1 else 0.0
    margins.append(delta)
    if delta < TAU: ambiguous_qids.append(did)

total_q = len(step3_baseline)
baseline_acc = baseline_correct / total_q * 100.0
print(f"\nBaseline OMGM Accuracy: {baseline_acc:.2f}% ({baseline_correct}/{total_q})")
print(f"Score Margin — Mean: {np.mean(margins):.4f} | Min: {np.min(margins):.4f} | Max: {np.max(margins):.4f}")
print(f"Ambiguous Queries (Δ < {TAU}): {len(ambiguous_qids)} / {total_q} ({len(ambiguous_qids)/total_q*100:.1f}%)")
print(f"Fast-Path Queries (Δ ≥ {TAU}): {total_q - len(ambiguous_qids)} / {total_q} ({(total_q - len(ambiguous_qids))/total_q*100:.1f}%)")



# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:43:43.416502Z","iopub.execute_input":"2026-10-22T06:43:43.416853Z","iopub.status.idle":"2026-10-22T06:44:00.686063Z","shell.execute_reply.started":"2026-10-22T06:43:43.416827Z","shell.execute_reply":"2026-10-22T06:44:00.685246Z"}}
# %% [markdown]
# ### Cell 8: Knowledge Base Loading

# %%
print("\n" + "=" * 70)
print("CELL 8: KNOWLEDGE BASE LOADING")
print("=" * 70)

t0 = time.time()
with open(KB_PATH, "r") as f: raw_kb = json.load(f)
kb_dict = {
    url: KnowledgeEntry(
        title=info.get("title", ""),
        url=url,
        section_titles=info.get("section_titles", []),
        section_texts=info.get("section_texts", []),
    )
    for url, info in raw_kb.items()
}
del raw_kb; gc.collect()
print(f"Loaded {len(kb_dict):,} Wikipedia articles in {time.time()-t0:.1f}s.")



# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:44:00.687276Z","iopub.execute_input":"2026-10-22T06:44:00.687620Z","iopub.status.idle":"2026-10-22T06:44:09.637151Z","shell.execute_reply.started":"2026-10-22T06:44:00.687589Z","shell.execute_reply":"2026-10-22T06:44:09.636366Z"}}
# %% [markdown]
# ### Cell 9: Pilot Run — Verify LoRA Verifier on 3 Ambiguous Queries (Visible Output)

# %%
print("\n" + "=" * 70)
print("CELL 9: PILOT RUN — FINE-TUNED VERIFIER ON 3 QUERIES")
print("=" * 70)

for pilot_did in ambiguous_qids[:3]:
    p_item = step3_baseline[pilot_did]
    s2_item = step2_results[pilot_did]
    cand1 = s2_item["retrieved_entities"][0]
    cand2 = s2_item["retrieved_entities"][1]
    title1, title2 = cand1.get("title", "?"), cand2.get("title", "?")
    delta = s2_item["retrieval_similarities"][0] - s2_item["retrieval_similarities"][1]
    img_id = sample_meta.get(pilot_did, {}).get("image_id") or p_item.get("dataset_image_ids", pilot_did)
    raw_img = get_query_image(img_id, QUERY_IMG_DIR)

    choice, debug = ask_visual_choice_finetuned(raw_img, title1, title2)

    print("-" * 65)
    print(f"Query ID   : {pilot_did}")
    print(f"Question   : {p_item.get('question', '')}")
    print(f"Δ (S1-S2)  : {delta:.4f}  [Ambiguous: Δ < {TAU}]")
    print(f"Candidate 1: {title1}")
    print(f"Candidate 2: {title2}")
    print(f"LoRA Choice: → {choice} ({debug})")
    print(f"Baseline Pred: '{p_item.get('prediction')}' | GT: '{p_item.get('ground_truth_answers')}'")
    print(f"Baseline Correct: {check_match(p_item.get('prediction',''), p_item.get('ground_truth_answers',''))}")
print("-" * 65)



# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T06:44:09.638865Z","iopub.execute_input":"2026-10-22T06:44:09.639241Z","iopub.status.idle":"2026-10-22T07:11:34.837270Z","shell.execute_reply.started":"2026-10-22T06:44:09.639175Z","shell.execute_reply":"2026-10-22T07:11:34.836542Z"}}
# %% [markdown]
# ### Cell 10: Main Agentic Controller Loop (Fine-Tuned Verifier)

# %%
print("\n" + "=" * 70)
print("CELL 10: AGENTIC CONTROLLER LOOP (FINE-TUNED LORA VERIFIER)")
print("=" * 70)
print(f"Processing {len(ambiguous_qids)} ambiguous queries using fine-tuned verifier...")

agentic_predictions = {qid: {**info, "agent_triggered": False, "agent_switched_to_b": False, "action_note": "Fast-Path Baseline"} for qid, info in step3_baseline.items()}

switched_b = 0
cat_c_refreshed = 0
loop_start = time.time()
checkpoint_path = os.path.join(OUTPUT_DIR, "finetuned_checkpoint.json")

for i, qid in enumerate(tqdm(ambiguous_qids, desc="Fine-Tuned Agentic Loop")):
    s2_data = step2_results[qid]
    b_info = step3_baseline[qid]
    question = b_info["question"]

    cands = s2_data.get("retrieved_entities", [])
    if len(cands) < 2: continue

    title1 = cands[0].get("title", "Candidate 1")
    title2 = cands[1].get("title", "Candidate 2")
    url1   = cands[0].get("url", "").strip()
    url2   = cands[1].get("url", "").strip()

    # Load query image
    img_id = sample_meta.get(qid, {}).get("image_id") or b_info.get("dataset_image_ids", qid)
    raw_img = get_query_image(img_id, QUERY_IMG_DIR)

    # Select best sections for both candidates
    sec1_text, score1 = select_best_section(question, kb_dict.get(url1))
    if not sec1_text: sec1_text = b_info.get("top1_section", "")
    sec2_text, score2 = select_best_section(question, kb_dict.get(url2))
    if not sec2_text:
        secs2 = reconstruct_wiki_sections(kb_dict[url2]) if kb_dict.get(url2) else []
        sec2_text = secs2[0] if secs2 else ""

    # Fine-tuned single-pass visual verification (no two-pass debiasing needed)
    choice, dec = ask_visual_choice_finetuned(raw_img, title1, title2)
    switch_to_b = (choice == "B") or (score2 > score1 + 2.0 and score2 > 0.0)

    new_pred = b_info["prediction"]
    action_note = "Kept Candidate 1 (Baseline Evidence)"

    if switch_to_b:
        switched_b += 1
        action_note = f"Switched to {title2} (Cat B LoRA: {dec})"
        new_pred = generate_answer(raw_img, question, sec2_text)
    elif sec1_text and sec1_text != b_info.get("top1_section", "") and score1 > 1.0:
        cat_c_refreshed += 1
        action_note = f"Refreshed Section for {title1} (Cat C BGE: {score1:.2f})"
        new_pred = generate_answer(raw_img, question, sec1_text)

    agentic_predictions[qid].update({
        "prediction": new_pred,
        "agent_triggered": True,
        "agent_switched_to_b": switch_to_b,
        "agent_score_margin": float(s2_data["retrieval_similarities"][0] - s2_data["retrieval_similarities"][1]),
        "agent_options": [title1, title2],
        "verifier_output": dec,
        "action_note": action_note,
    })

    if (i + 1) % 50 == 0:
        with open(checkpoint_path, "w") as f: json.dump(agentic_predictions, f, indent=2)
        elapsed = time.time() - loop_start
        print(f"  Checkpoint @ {i+1}/{len(ambiguous_qids)} | Elapsed: {elapsed/60:.1f} min | Switches: {switched_b}")

total_loop_time = time.time() - loop_start
print(f"\nAgentic Loop Finished in {total_loop_time:.1f}s ({total_loop_time/len(ambiguous_qids):.2f}s per query).")
print(f"Visual switches to Candidate 2: {switched_b}")
print(f"Category C section refreshes:   {cat_c_refreshed}")



# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T07:11:34.838572Z","iopub.execute_input":"2026-10-22T07:11:34.838911Z","iopub.status.idle":"2026-10-22T07:11:34.948301Z","shell.execute_reply.started":"2026-10-22T07:11:34.838889Z","shell.execute_reply":"2026-10-22T07:11:34.947677Z"}}
# %% [markdown]
# ### Cell 11: Final Zero-Leakage Evaluation & Comparative Results Table

# %%
print("\n" + "=" * 70)
print("CELL 11: FINAL ZERO-LEAKAGE EVALUATION (FINE-TUNED CONTROLLER)")
print("=" * 70)

final_correct = 0
baseline_correct_final = 0
recovered_events, degraded_events = [], []
preserved_correct = 0

for qid in agentic_predictions:
    b_info = step3_baseline[qid]
    a_info = agentic_predictions[qid]
    gt = a_info.get("ground_truth_answers") or sample_meta.get(qid, {}).get("ground_truth_answers", "")
    was_right = check_match(b_info.get("prediction", ""), gt)
    now_right = check_match(a_info.get("prediction", ""), gt)
    a_info["is_correct"] = now_right
    if was_right: baseline_correct_final += 1
    if now_right: final_correct += 1
    if not was_right and now_right:
        recovered_events.append({"qid": qid, "question": b_info.get("question",""), "action": a_info.get("action_note",""), "old_pred": b_info.get("prediction",""), "new_pred": a_info.get("prediction",""), "gt": gt})
    elif was_right and not now_right:
        degraded_events.append({"qid": qid, "question": b_info.get("question",""), "action": a_info.get("action_note",""), "old_pred": b_info.get("prediction",""), "new_pred": a_info.get("prediction",""), "gt": gt})
    elif was_right and now_right:
        preserved_correct += 1

final_acc   = final_correct / total_q * 100.0
base_acc    = baseline_correct_final / total_q * 100.0
delta_acc   = final_acc - base_acc
net_rec     = final_correct - baseline_correct_final
avg_calls   = 1.0 + (len(ambiguous_qids) / total_q)

print(f"\n{'Metric':<35} | {'Baseline (OMGM)':<20} | {'Ours (LoRA Verifier)':<20} | Delta")
print("-" * 90)
print(f"{'Overall VQA Accuracy':<35} | {base_acc:6.2f}% ({baseline_correct_final}/{total_q}) | {final_acc:6.2f}% ({final_correct}/{total_q})   | {delta_acc:+.2f}% ({net_rec:+d})")
print(f"{'Ambiguous Queries Triggered':<35} | {'N/A':<20} | {len(ambiguous_qids):<20} | {len(ambiguous_qids)/total_q*100:.1f}%")
print(f"{'Gross Recoveries (Wrong→Right)':<35} | {'0 (rigid)':<20} | {len(recovered_events):<20} | +{len(recovered_events)}")
print(f"{'Degradations (Right→Wrong)':<35} | {'0 (rigid)':<20} | {len(degraded_events):<20} | -{len(degraded_events)}")
print(f"{'Net Improvement':<35} | {'0':<20} | {net_rec:+d}{' queries':<19} | Honest Net")
print(f"{'Avg. Generator Calls/Query':<35} | {'1.00':<20} | {avg_calls:.2f}{'':18} | +{avg_calls-1:.2f} overhead")
print("=" * 90)



# %% [code] {"execution":{"iopub.status.busy":"2026-10-22T07:11:34.949320Z","iopub.execute_input":"2026-10-22T07:11:34.949660Z","iopub.status.idle":"2026-10-22T07:11:34.993494Z","shell.execute_reply.started":"2026-10-22T07:11:34.949620Z","shell.execute_reply":"2026-10-22T07:11:34.992663Z"}}
# %% [markdown]
# ### Cell 12: Save All Results & Download Package

# %%
print("\n" + "=" * 70)
print("CELL 12: SAVING ALL RESULTS")
print("=" * 70)

summary = {
    "model": "LLaVA-1.5-7B + LoRA Verifier (Fine-Tuned, Symmetric 6k Pairs)",
    "lora_adapter": LORA_ADAPTER,
    "tau": TAU,
    "total_queries": total_q,
    "baseline_accuracy": round(base_acc, 2),
    "agentic_accuracy": round(final_acc, 2),
    "accuracy_improvement": round(delta_acc, 2),
    "net_recovered_queries": net_rec,
    "gross_recovered_queries": len(recovered_events),
    "degraded_queries": len(degraded_events),
    "preserved_correct_queries": preserved_correct,
    "ambiguous_queries_triggered": len(ambiguous_qids),
    "trigger_rate_pct": round(len(ambiguous_qids) / total_q * 100.0, 2),
    "switched_to_b_count": switched_b,
    "cat_c_refreshed_count": cat_c_refreshed,
    "average_calls_per_query": round(avg_calls, 2),
    "loop_time_minutes": round(total_loop_time / 60, 2),
}

with open(f"{OUTPUT_DIR}/finetuned_results_summary.json", "w") as f: json.dump(summary, f, indent=2)
with open(f"{OUTPUT_DIR}/finetuned_final_predictions.json", "w") as f: json.dump(agentic_predictions, f, indent=2)
with open(f"{OUTPUT_DIR}/finetuned_recovered_queries.json", "w") as f:
    json.dump({"recovered_events": recovered_events, "degraded_events": degraded_events}, f, indent=2)

print(f"Saved to '{OUTPUT_DIR}/':")
for fn in os.listdir(OUTPUT_DIR):
    sz = os.path.getsize(os.path.join(OUTPUT_DIR, fn)) / 1024
    print(f"  - {fn} ({sz:.1f} KB)")
print("\nDone! Download these files from Kaggle's Output tab.")

# %% [markdown]
# ### Cell 13: Category B + C Agentic Controller (Multi-Section Context Expansion)
# Extends the fine-tuned 7B controller to address Category C (Section Selection Errors)
# by concatenating Top-1 and Top-2 Wikipedia sections when Candidate 1 is verified.

# %%
print("\n" + "=" * 70)
print("CELL 13: CATEGORY B + C CONTROLLER (MULTI-SECTION CONTEXT EXPANSION - 7B)")
print("=" * 70)

def select_top2_sections_combined(question: str, entry: Optional[KnowledgeEntry], max_chars: int = 3200) -> Tuple[str, str]:
    """
    Returns (top1_section, combined_top2_sections).
    Combines Top-1 and Top-2 sections to solve Category C (Section Selection Errors: 203 queries).
    """
    if not entry:
        return "", ""
    sections = reconstruct_wiki_sections(entry)
    if not sections:
        return "", ""
    if len(sections) == 1:
        return sections[0], sections[0]

    # Rank sections using BGE sequence classification
    if bge_model is not None and torch.cuda.is_available():
        try:
            pairs = [[question, s[:1200]] for s in sections]
            inp = bge_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
            with torch.no_grad():
                scores = bge_model(**inp).logits.view(-1).float().cpu().tolist()
            ranked_indices = np.argsort(scores)[::-1]
            top1 = sections[ranked_indices[0]]
            top2 = sections[ranked_indices[1]] if len(ranked_indices) > 1 else ""

            combined = f"{top1}\n\n--- [Additional Section Evidence] ---\n\n{top2}" if top2 else top1
            return top1, combined[:max_chars]
        except Exception:
            pass

    # Lexical fallback ranking
    stopwords = {"what", "is", "the", "of", "this", "in", "how", "many", "a", "an", "to", "for", "does", "do", "which", "where", "who", "when", "why"}
    qw = [w.lower() for w in re.findall(r"\b\w+\b", question) if w.lower() not in stopwords]
    scored_secs = []
    for sec in sections:
        sl = sec.lower()
        sc = sum(5.0 if w in sl.split("\n")[1] else 1.0 if w in sl else 0.0 for w in qw)
        scored_secs.append((sc, sec))
    scored_secs.sort(key=lambda x: x[0], reverse=True)
    top1 = scored_secs[0][1]
    top2 = scored_secs[1][1] if len(scored_secs) > 1 else ""
    combined = f"{top1}\n\n--- [Additional Section Evidence] ---\n\n{top2}" if top2 else top1
    return top1, combined[:max_chars]

# Checkpoint support for Category B+C
checkpoint_bc_path = os.path.join(OUTPUT_DIR, "finetuned_7b_cat_bc_checkpoint.json")

if os.path.exists(checkpoint_bc_path):
    print(f"Found existing Category B+C checkpoint at '{checkpoint_bc_path}'. Resuming...")
    with open(checkpoint_bc_path, "r") as f:
        cat_bc_predictions = json.load(f)
    print(f"Resumed {len(cat_bc_predictions):,} entries.")
else:
    cat_bc_predictions = {
        qid: {
            **info,
            "agent_triggered": False,
            "agent_switched_to_b": False,
            "action_note": "Fast-Path Baseline"
        }
        for qid, info in step3_baseline.items()
    }

switched_b_count = sum(1 for p in cat_bc_predictions.values() if p.get("agent_switched_to_b", False))
cat_c_expanded_count = sum(1 for p in cat_bc_predictions.values() if "Expanded Top-2 Sections" in p.get("action_note", ""))
loop_start = time.time()

remaining_ambiguous = [qid for qid in ambiguous_qids if not cat_bc_predictions[qid].get("agent_triggered", False)]
print(f"Processing {len(remaining_ambiguous)} / {len(ambiguous_qids)} remaining ambiguous queries for Category B+C...")

for i, qid in enumerate(tqdm(remaining_ambiguous, desc="Fine-Tuned 7B Cat B+C Loop")):
    s2_data = step2_results[qid]
    b_info = step3_baseline[qid]
    question = b_info["question"]

    cands = s2_data.get("retrieved_entities", [])
    if len(cands) < 2:
        continue

    title1 = cands[0].get("title", "Candidate 1")
    title2 = cands[1].get("title", "Candidate 2")
    url1   = cands[0].get("url", "").strip()
    url2   = cands[1].get("url", "").strip()

    # Load query image
    img_id = sample_meta.get(qid, {}).get("image_id") or b_info.get("dataset_image_ids", qid)
    raw_img = get_query_image(img_id, QUERY_IMG_DIR)

    # 1. Single-pass fine-tuned visual verification (Category B)
    choice, dec = ask_visual_choice_finetuned(raw_img, title1, title2)
    switch_to_b = (choice == "B")

    if switch_to_b:
        # Category B: Switch to Candidate 2 with its top-2 sections
        switched_b_count += 1
        _, sec2_combined = select_top2_sections_combined(question, kb_dict.get(url2))
        if not sec2_combined:
            sec2_text, _ = select_best_section(question, kb_dict.get(url2))
            sec2_combined = sec2_text
        action_note = f"Switched to {title2} (Cat B LoRA: {dec})"
        new_pred = generate_answer(raw_img, question, sec2_combined, max_chars=3200)
    else:
        # Category C: Candidate 1 verified, but expand to Top-1 + Top-2 sections
        cat_c_expanded_count += 1
        _, sec1_combined = select_top2_sections_combined(question, kb_dict.get(url1))
        if not sec1_combined:
            sec1_combined = b_info.get("top1_section", "")
        action_note = f"Kept {title1} + Expanded Top-2 Sections (Cat C)"
        new_pred = generate_answer(raw_img, question, sec1_combined, max_chars=3200)

    cat_bc_predictions[qid].update({
        "prediction": new_pred,
        "agent_triggered": True,
        "agent_switched_to_b": switch_to_b,
        "action_note": action_note,
        "verifier_output": dec,
    })

    if (i + 1) % 50 == 0 or (i + 1) == len(remaining_ambiguous):
        with open(checkpoint_bc_path, "w") as f:
            json.dump(cat_bc_predictions, f, indent=2)
        elapsed = time.time() - loop_start
        print(f"  Checkpoint @ {i+1}/{len(remaining_ambiguous)} | Elapsed: {elapsed/60:.1f} min | Switches: {switched_b_count} | Expansions: {cat_c_expanded_count}")

total_time = time.time() - loop_start
print(f"\nCategory B+C Controller Finished in {total_time/60:.1f} minutes ({total_time/max(1, len(remaining_ambiguous)):.2f}s per query).")
print(f"  Switches to Candidate 2 (Cat B) : {switched_b_count}")
print(f"  Multi-Section Expansions (Cat C): {cat_c_expanded_count}")

# Save Category B+C predictions
os.makedirs(OUTPUT_DIR, exist_ok=True)
bc_pred_path = os.path.join(OUTPUT_DIR, "finetuned_7b_cat_bc_predictions.json")
with open(bc_pred_path, "w") as f:
    json.dump(cat_bc_predictions, f, indent=2)
print(f"Saved: {bc_pred_path}")


# %% [markdown]
# ### Cell 14: Category B + C Scorecard & Statistical Evaluation (7B)

# %%
print("\n" + "=" * 70)
print("CELL 14: CATEGORY B + C SCORECARD & EVALUATION (7B)")
print("=" * 70)

base_correct = 0
agent_correct = 0
rec_events = []
deg_events = []

for qid, b_item in step3_baseline.items():
    a_item = cat_bc_predictions[qid]
    gt = b_item.get("ground_truth_answers", "")

    b_match = check_match(b_item.get("prediction", ""), gt)
    a_match = check_match(a_item.get("prediction", ""), gt)
    action  = a_item.get("action_note", "")
    cat     = "Category B (Candidate 2)" if "Switched to" in action else "Category C (Multi-Section)"

    if b_match:
        base_correct += 1
    if a_match:
        agent_correct += 1

    if not b_match and a_match:
        rec_events.append({
            "qid": qid,
            "category": cat,
            "question": a_item["question"],
            "action": action,
            "old_pred": b_item["prediction"],
            "new_pred": a_item["prediction"],
            "gt": gt
        })
    elif b_match and not a_match:
        deg_events.append({
            "qid": qid,
            "category": cat,
            "question": a_item["question"],
            "action": action,
            "old_pred": b_item["prediction"],
            "new_pred": a_item["prediction"],
            "gt": gt
        })

total_q   = len(step3_baseline)
base_acc  = base_correct / total_q * 100.0
agent_acc = agent_correct / total_q * 100.0
net_gain  = len(rec_events) - len(deg_events)

rec_b = sum(1 for e in rec_events if "Category B" in e["category"])
rec_c = sum(1 for e in rec_events if "Category C" in e["category"])
deg_b = sum(1 for e in deg_events if "Category B" in e["category"])
deg_c = sum(1 for e in deg_events if "Category C" in e["category"])

# McNemar's test with continuity correction
b_count = len(deg_events)  # Baseline right, Agent wrong
c_count = len(rec_events)  # Baseline wrong, Agent right
n_discordant = b_count + c_count
if n_discordant > 0:
    mcnemar_chi2 = ((abs(b_count - c_count) - 1.0) ** 2) / n_discordant
    import scipy.stats as stats
    try:
        mcnemar_p = stats.chi2.sf(mcnemar_chi2, df=1)
    except Exception:
        import math
        mcnemar_p = math.erfc(math.sqrt(mcnemar_chi2) / math.sqrt(2.0))
else:
    mcnemar_chi2 = 0.0
    mcnemar_p = 1.0

win_rate = (len(rec_events) / n_discordant * 100) if n_discordant > 0 else 0.0

print(f"\n{'Metric':<38} | {'Baseline (OMGM)':<20} | {'Ours (Cat B+C 7B)':<20} | {'Delta':<15}")
print("-" * 100)
print(f"{'Overall VQA Accuracy':<38} | {base_acc:.2f}% ({base_correct}/{total_q}){'':<4} | {agent_acc:.2f}% ({agent_correct}/{total_q}){'':<4} | +{agent_acc - base_acc:+.2f}% (+{net_gain})")
print(f"{'Gross Recoveries (Wrong->Right)':<38} | {'0 (rigid)':<20} | {len(rec_events):<20} | +{len(rec_events)}")
print(f"{'  - Category B Recoveries (Cand 2)':<38} | {'--':<20} | {rec_b:<20} | +{rec_b}")
print(f"{'  - Category C Recoveries (Multi-Sec)':<38} | {'--':<20} | {rec_c:<20} | +{rec_c}")
print(f"{'Degradations (Right->Wrong)':<38} | {'0 (rigid)':<20} | {len(deg_events):<20} | -{len(deg_events)}")
print(f"{'  - Category B Degradations (Cand 2)':<38} | {'--':<20} | {deg_b:<20} | -{deg_b}")
print(f"{'  - Category C Degradations (Multi-Sec)':<38} | {'--':<20} | {deg_c:<20} | -{deg_c}")
print(f"{'Net Gain from Category B':<38} | {'--':<20} | {rec_b - deg_b:+d} queries{'':<10} | Cat B Net")
print(f"{'Net Gain from Category C':<38} | {'--':<20} | {rec_c - deg_c:+d} queries{'':<10} | Cat C Net")
print(f"{'Overall Net Improvement':<38} | {'0':<20} | {net_gain:+d} queries{'':<10} | Honest Net")
print(f"{'Decision Win-Rate on Flips':<38} | {'--':<20} | {win_rate:.2f}%{'':<14} | ({len(rec_events)}/{n_discordant})")
print(f"{'McNemar Chi-Squared (p-val)':<38} | {'--':<20} | {mcnemar_chi2:.3f} (p={mcnemar_p:.4f}) | Significance")
print("=" * 100)

# Save structured summary and recovery audit JSONs
summary_bc = {
    "model": "LLaVA-1.5-7B + LoRA Verifier + Multi-Section Expansion (Category B+C)",
    "lora_adapter": LORA_ADAPTER,
    "tau": TAU,
    "total_queries": total_q,
    "baseline_accuracy": round(base_acc, 2),
    "agentic_accuracy": round(agent_acc, 2),
    "accuracy_improvement": round(agent_acc - base_acc, 2),
    "net_recovered_queries": net_gain,
    "gross_recovered_queries": len(rec_events),
    "gross_recovered_cat_b": rec_b,
    "gross_recovered_cat_c": rec_c,
    "degraded_queries": len(deg_events),
    "degraded_cat_b": deg_b,
    "degraded_cat_c": deg_c,
    "net_cat_b": rec_b - deg_b,
    "net_cat_c": rec_c - deg_c,
    "decision_win_rate_pct": round(win_rate, 2),
    "mcnemar_chi2": round(mcnemar_chi2, 4),
    "mcnemar_p_value": round(mcnemar_p, 5),
    "total_ambiguous_triggered": len(ambiguous_qids),
    "switched_to_b_count": switched_b_count,
    "multi_section_expanded_count": cat_c_expanded_count,
    "runtime_minutes": round(total_time / 60, 2)
}

summary_bc_path = os.path.join(OUTPUT_DIR, "finetuned_7b_cat_bc_results_summary.json")
with open(summary_bc_path, "w") as f:
    json.dump(summary_bc, f, indent=2)

rec_bc_path = os.path.join(OUTPUT_DIR, "finetuned_7b_cat_bc_recovered_queries.json")
with open(rec_bc_path, "w") as f:
    json.dump({"recovered_events": rec_events, "degraded_events": deg_events}, f, indent=2)

print(f"\nSaved Category B+C artifacts:")
print(f"  - {bc_pred_path}")
print(f"  - {summary_bc_path}")
print(f"  - {rec_bc_path}")

