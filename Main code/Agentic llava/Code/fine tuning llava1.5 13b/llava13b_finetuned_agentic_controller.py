# %% [markdown]
# # =============================================================================
# # Fine-Tuned LLaVA-1.5-13B Agentic Controller for KB-VQA (InfoSeek)
# # =============================================================================
# # Architecture:
# #   - Base Model: LLaVA-1.5-13B (llava-hf/llava-1.5-13b-hf) in 4-bit NF4
# #   - LoRA Adapter: Fine-tuned on symmetric 6,000 entity pairs (bias < 2.7%)
# #   - Decoupled Inference:
# #       * `ask_visual_choice_finetuned()`: Single-pass verification via LoRA adapter
# #       * `generate_answer()`: Pure base model generation via `disable_adapter()`
# #   - Zero-Leakage Intrinsic Score Margin Gating (tau = 0.025):
# #       * Fast-Path (Delta >= tau): Retains baseline prediction (0 extra compute)
# #       * Ambiguity Zone (Delta < tau): Single-pass fine-tuned visual verification
# #   - Checkpoint Auto-Resume: Recovers seamlessly if session is interrupted
# #
# # Verified LoRA Adapter Path:
# #   /kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/llava_13b_verifier_lora
# # =============================================================================

# %% [markdown]
# ### Cell 0: Package Installations (Kaggle Environment Setup)

# %%
# In a Kaggle notebook cell, uncomment and run if needed:
# !pip install -q -U bitsandbytes accelerate peft

import subprocess
import sys
from typing import Optional

def ensure_package(pkg_name: str, import_name: Optional[str] = None):
    imp = import_name or pkg_name
    try:
        __import__(imp)
    except ImportError:
        print(f"Installing {pkg_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", pkg_name])

ensure_package("bitsandbytes")
ensure_package("accelerate")
ensure_package("peft")
print("All dependencies ready.")

# %% [markdown]
# ### Cell 1: Environment Setup, Seeds & Path Configuration

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
from tqdm.auto import tqdm

print("=" * 70)
print("CELL 1: ENVIRONMENT SETUP, SEEDS & PATH CONFIGURATION (13B)")
print("=" * 70)

# Deterministic Reproducibility
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Deterministic seed set to {SEED}.")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} CUDA Device(s):")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU [{i}]: {props.name} | VRAM: {props.total_memory / 1024**3:.2f} GB")

# Path Configuration & Autodetection
BASE_DS_DIR = "/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds"
ALT_DS_DIR  = "/kaggle/input/agentic-vqa-ds"
LOCAL_PRE   = "pre helper"

def resolve_path(candidates: List[str], is_dir: bool = False) -> str:
    for p in candidates:
        if os.path.exists(p):
            if is_dir and os.path.isdir(p):
                return p
            elif not is_dir and os.path.isfile(p):
                return p
            elif os.path.exists(p):
                return p
    return candidates[0]

STEP2_PATH = resolve_path([
    f"{BASE_DS_DIR}/result_step_2.json",
    f"{BASE_DS_DIR}/result_step2.json",
    f"{ALT_DS_DIR}/result_step_2.json",
    f"{LOCAL_PRE}/res/result_step2.json"
])

STEP3_PATH = resolve_path([
    f"{BASE_DS_DIR}/step3_predictions.json",
    f"{ALT_DS_DIR}/step3_predictions.json",
    f"{LOCAL_PRE}/res/step3_predictions.json"
])

SAMPLE_PATH = resolve_path([
    f"{BASE_DS_DIR}/infoseek_test_sample_1000.csv",
    f"{ALT_DS_DIR}/infoseek_test_sample_1000.csv",
    f"{LOCAL_PRE}/files/infoseek_test_sample_1000.csv"
])

KB_PATH = resolve_path([
    f"{BASE_DS_DIR}/infoseek_100k_wiki/wiki_100_dict_v4.json",
    f"{ALT_DS_DIR}/infoseek_100k_wiki/wiki_100_dict_v4.json"
])

QUERY_IMG_DIR = resolve_path([
    f"{BASE_DS_DIR}/infoseek_val_1000_images/infoseek_val_1000_images",
    f"{BASE_DS_DIR}/infoseek_val_1000_images",
    f"{ALT_DS_DIR}/infoseek_val_1000_images/infoseek_val_1000_images"
], is_dir=True)

LORA_ADAPTER = resolve_path([
    f"{BASE_DS_DIR}/llava_13b_verifier_lora",
    f"{ALT_DS_DIR}/llava_13b_verifier_lora",
    "/kaggle/input/llava_13b_verifier_lora",
    "/kaggle/working/llava_1_5_13b_agentic_verifier_lora"
], is_dir=True)

OUTPUT_DIR = "/kaggle/working/agentic_results_finetuned_13b"
TAU = 0.025

print(f"\nResolved Dataset & Model Paths:")
print(f"  Step 2 Result  : {STEP2_PATH}  (exists: {os.path.exists(STEP2_PATH)})")
print(f"  Step 3 Baseline: {STEP3_PATH}  (exists: {os.path.exists(STEP3_PATH)})")
print(f"  Sample CSV     : {SAMPLE_PATH}  (exists: {os.path.exists(SAMPLE_PATH)})")
print(f"  KB JSON        : {KB_PATH}  (exists: {os.path.exists(KB_PATH)})")
print(f"  Query Images   : {QUERY_IMG_DIR}  (exists: {os.path.isdir(QUERY_IMG_DIR)})")
print(f"  LoRA 13B Adapter: {LORA_ADAPTER}  (exists: {os.path.isdir(LORA_ADAPTER)})")
print(f"  Tau Threshold  : {TAU}")
print(f"  Output Dir     : {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.exists(STEP2_PATH), f"Step 2 file missing: {STEP2_PATH}"
assert os.path.exists(STEP3_PATH), f"Step 3 baseline missing: {STEP3_PATH}"
assert os.path.exists(SAMPLE_PATH), f"Sample CSV missing: {SAMPLE_PATH}"
assert os.path.exists(KB_PATH), f"Knowledge base missing: {KB_PATH}"
assert os.path.isdir(QUERY_IMG_DIR), f"Query image directory missing: {QUERY_IMG_DIR}"
assert os.path.isdir(LORA_ADAPTER), f"LoRA 13B adapter directory missing: {LORA_ADAPTER}"

# %% [markdown]
# ### Cell 2: Evaluation Metric Helpers (Canonical OMGM `check_match`)

# %%
print("\n" + "=" * 70)
print("CELL 2: INFOSEEK EVALUATION METRIC HELPERS")
print("=" * 70)

def normalize_answer(text: Any) -> str:
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.lower().split())

def extract_number(text: Any) -> Optional[float]:
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(text).replace(",", ""))
    try:
        return float(matches[0]) if matches else None
    except ValueError:
        return None

def metric_numerical_range(pred: float, ans: float, tol: float = 0.10) -> bool:
    lo = ans * (1 - tol) if ans >= 0 else ans * (1 + tol)
    hi = ans * (1 + tol) if ans >= 0 else ans * (1 - tol)
    return min(lo, hi) <= pred <= max(lo, hi)

def check_match(pred: str, gt: Any) -> bool:
    if not pred or not gt:
        return False
    np_ = normalize_answer(pred)
    if not np_:
        return False
    gt_list = gt if isinstance(gt, list) else [g.strip() for g in str(gt).split("|") if str(g).strip()]
    for g in gt_list:
        ng = normalize_answer(g)
        if ng and (ng == np_ or f" {ng} " in f" {np_} " or ng in np_):
            return True
    pn = extract_number(pred)
    if pn is not None:
        for g in gt_list:
            c = str(g).strip().replace(",", "")
            if re.fullmatch(r"[-+]?\d*\.?\d+", c):
                if metric_numerical_range(pn, float(c), tol=0.10):
                    return True
    return False

print("Evaluation metric loaded: Canonical OMGM check_match with +/- 10% numeric tolerance.")

# %% [markdown]
# ### Cell 3: Knowledge Base & Query Image Helpers

# %%
print("\n" + "=" * 70)
print("CELL 3: KNOWLEDGE BASE & QUERY IMAGE HELPERS")
print("=" * 70)

class KnowledgeEntry:
    def __init__(self, title: str, url: str, section_titles: List[str], section_texts: List[str]):
        self.title = title
        self.url = url
        self.section_titles = section_titles
        self.section_texts = section_texts

def reconstruct_wiki_sections(entry: KnowledgeEntry) -> List[str]:
    sections = []
    for i, sec_t in enumerate(entry.section_titles):
        sl = sec_t.lower()
        if "external links" in sl or "references" in sl:
            continue
        text = entry.section_texts[i] if i < len(entry.section_texts) else ""
        sections.append(f"# Wiki Article: {entry.title}\n## Section Title: {sec_t}\n{text}")
    return sections

def get_query_image(image_id: str, search_dir: str) -> PIL.Image.Image:
    clean = str(image_id).split("|")[0].strip()
    for ext in ["", ".jpg", ".jpeg", ".JPEG", ".png"]:
        p = os.path.join(search_dir, f"{clean}{ext}")
        if os.path.isfile(p):
            try:
                return PIL.Image.open(p).convert("RGB")
            except Exception:
                pass
    for m in glob.glob(os.path.join(search_dir, f"*{clean}*")):
        if os.path.isfile(m) and m.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                return PIL.Image.open(m).convert("RGB")
            except Exception:
                pass
    return PIL.Image.new("RGB", (224, 224), color=(128, 128, 128))

print("KnowledgeEntry class and image resolution helpers loaded.")

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
    if not entry:
        return "", -999.0
    sections = reconstruct_wiki_sections(entry)
    if not sections:
        return "", -999.0
    if len(sections) == 1:
        return sections[0], 0.0
    if bge_model is not None and torch.cuda.is_available():
        try:
            pairs = [[question, s[:1500]] for s in sections]
            inp = bge_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
            with torch.no_grad():
                scores = bge_model(**inp).logits.view(-1).float().cpu().tolist()
            best = int(np.argmax(scores))
            return sections[best], float(scores[best])
        except Exception:
            pass
    # Lexical fallback
    stopwords = {"what", "is", "the", "of", "this", "in", "how", "many", "a", "an", "to", "for", "does", "do", "which", "where", "who", "when", "why"}
    qw = [w.lower() for w in re.findall(r"\b\w+\b", question) if w.lower() not in stopwords]
    best_sec, best_sc = sections[0], -1.0
    for sec in sections:
        sl = sec.lower()
        sc = sum(5.0 if w in sl.split("\n")[1] else 1.0 if w in sl else 0.0 for w in qw)
        if sc > best_sc:
            best_sc, best_sec = sc, sec
    return best_sec, float(best_sc)

init_bge_reranker("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ### Cell 5: LLaVA-1.5-13B Base Model + LoRA Adapter Loading
# Loads base LLaVA-1.5-13B in 4-bit NF4 across dual GPUs (~4.0 GB per GPU).
# Attaches the fine-tuned 13B LoRA adapter (~248 MB deltas).
# Answers are generated using `disable_adapter()`, while entity choices use the active LoRA adapter.

# %%
print("\n" + "=" * 70)
print("CELL 5: LLaVA-1.5-13B BASE MODEL + LoRA ADAPTER LOADING")
print("=" * 70)

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "llava-hf/llava-1.5-13b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"1. Loading processor from '{MODEL_ID}'...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.unk_token
processor.tokenizer.padding_side = "right"

print("2. Loading base LLaVA-1.5-13B in 4-bit NF4...")
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

print(f"\n3. Attaching Fine-Tuned 13B LoRA Adapter from: '{LORA_ADAPTER}'")
assert os.path.isdir(LORA_ADAPTER), f"LoRA adapter directory not found: {LORA_ADAPTER}"
verifier_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
verifier_model.eval()
print("   LoRA 13B adapter attached successfully.")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  After LoRA attach — GPU [{i}]: {alloc:.2f} GB allocated")

print("\n4. Verification: Confirm LoRA 13B Adapter is Active (Inference Mode)")
cfg = list(verifier_model.peft_config.values())[0]
lora_r = cfg.r
lora_target = cfg.target_modules
print(f"   Active adapter(s): {list(verifier_model.peft_config.keys())}")
print(f"   Adapter type     : {cfg.peft_type}")
print(f"   LoRA rank (r)    : {lora_r}")
print(f"   LoRA target mods : {lora_target}")

adapter_weights_file = os.path.join(LORA_ADAPTER, "adapter_model.safetensors")
print(f"   Adapter weights  : {adapter_weights_file} (exists: {os.path.exists(adapter_weights_file)})")
assert os.path.exists(adapter_weights_file), "CRITICAL: adapter_model.safetensors missing!"
print("   NOTE: requires_grad=False is CORRECT for inference — LoRA hooks are active.")
print("   NOTE: generate_answer() will use disable_adapter() to restore pure base model.")

# %% [markdown]
# ### Cell 6: Inference Helper Functions
# `ask_visual_choice_finetuned` — Single-pass visual verifier using the fine-tuned 13B adapter.
# `generate_answer` — Generates answer using pure base model via `disable_adapter()`.

# %%
print("\n" + "=" * 70)
print("CELL 6: INFERENCE HELPER FUNCTIONS (13B)")
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
    """
    Generate factual answer using pure base model (OMGM-style encyclopedic VQA).
    Uses disable_adapter() to bypass the LoRA verifier for this call, maintaining
    exact parity with baseline generation.
    """
    trimmed = section_text[:max_chars]
    prompt_text = OMGM_INSTRUCTION + f"- Context: {trimmed}\n- Question: {question}\nJust answer the questions , no explanations needed. Short answer is:"
    full_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
    inputs = processor(images=img, text=full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        with verifier_model.disable_adapter():
            out = verifier_model.generate(**inputs, max_new_tokens=32, do_sample=False)
    plen = inputs.input_ids.shape[1]
    return processor.decode(out[0][plen:], skip_special_tokens=True).strip()

def ask_visual_choice_finetuned(img: PIL.Image.Image, title1: str, title2: str) -> Tuple[str, str]:
    """
    Single-pass fine-tuned 13B verifier.
    The LoRA adapter (trained on 50/50 symmetric pairs) has primacy bias < 2.7%,
    so a single forward pass is accurate, calibrated, and unbiased.
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
        out = verifier_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    plen = inputs.input_ids.shape[1]
    generated = processor.decode(out[0][plen:], skip_special_tokens=True).strip()
    gen_lower = generated.lower()

    if gen_lower.startswith("(b)") or gen_lower.startswith("b)") or gen_lower.startswith("b ") or gen_lower == "b":
        return "B", f"FineTuned13B→Cand2 | gen: {generated}"
    return "A", f"FineTuned13B→Cand1 | gen: {generated}"

print("Inference helpers loaded:")
print("  - generate_answer()            -> base_model (no LoRA, OMGM-style VQA)")
print("  - ask_visual_choice_finetuned() -> verifier_model (13B base + LoRA, single-pass)")

# %% [markdown]
# ### Cell 7: Data Loading & Baseline Analysis

# %%
print("\n" + "=" * 70)
print("CELL 7: DATA LOADING & BASELINE ANALYSIS")
print("=" * 70)

with open(STEP2_PATH, "r") as f:
    step2_results = json.load(f)
with open(STEP3_PATH, "r") as f:
    step3_baseline = json.load(f)

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
    if check_match(pinfo.get("prediction", ""), gt):
        baseline_correct += 1
    s2 = step2_results.get(did, {})
    scores = s2.get("retrieval_similarities", [])
    delta = float(scores[0] - scores[1]) if len(scores) > 1 else 0.0
    margins.append(delta)
    if delta < TAU:
        ambiguous_qids.append(did)

total_q = len(step3_baseline)
baseline_acc = baseline_correct / total_q * 100.0
print(f"\nBaseline OMGM Accuracy: {baseline_acc:.2f}% ({baseline_correct}/{total_q})")
print(f"Score Margin — Mean: {np.mean(margins):.4f} | Min: {np.min(margins):.4f} | Max: {np.max(margins):.4f}")
print(f"Ambiguous Queries (Δ < {TAU}): {len(ambiguous_qids)} / {total_q} ({len(ambiguous_qids)/total_q*100:.1f}%)")
print(f"Fast-Path Queries (Δ ≥ {TAU}): {total_q - len(ambiguous_qids)} / {total_q} ({(total_q - len(ambiguous_qids))/total_q*100:.1f}%)")

# %% [markdown]
# ### Cell 8: Knowledge Base Loading

# %%
print("\n" + "=" * 70)
print("CELL 8: KNOWLEDGE BASE LOADING")
print("=" * 70)

t0 = time.time()
with open(KB_PATH, "r") as f:
    raw_kb = json.load(f)

kb_dict = {
    url: KnowledgeEntry(
        title=info.get("title", ""),
        url=url,
        section_titles=info.get("section_titles", []),
        section_texts=info.get("section_texts", []),
    )
    for url, info in raw_kb.items()
}
del raw_kb
gc.collect()
print(f"Loaded {len(kb_dict):,} Wikipedia articles in {time.time()-t0:.1f}s.")

# %% [markdown]
# ### Cell 9: Pilot Run — Verify LoRA 13B Verifier on 3 Ambiguous Queries

# %%
print("\n" + "=" * 70)
print("CELL 9: PILOT RUN — FINE-TUNED 13B VERIFIER ON 3 QUERIES")
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
    print(f"LoRA Choice: -> {choice} ({debug})")
    print(f"Baseline Pred: '{p_item.get('prediction')}' | GT: '{p_item.get('ground_truth_answers')}'")
    print(f"Baseline Correct: {check_match(p_item.get('prediction',''), p_item.get('ground_truth_answers',''))}")
print("-" * 65)

# %% [markdown]
# ### Cell 10: Main Agentic Controller Loop (Fine-Tuned 13B Verifier)
# Features auto-resume from checkpoint if session interrupted.

# %%
print("\n" + "=" * 70)
print("CELL 10: AGENTIC CONTROLLER LOOP (FINE-TUNED 13B LoRA VERIFIER)")
print("=" * 70)

checkpoint_path = os.path.join(OUTPUT_DIR, "finetuned_13b_checkpoint.json")

# Initialize predictions dictionary (supports resuming from checkpoint)
if os.path.exists(checkpoint_path):
    print(f"Found existing checkpoint at '{checkpoint_path}'. Resuming...")
    with open(checkpoint_path, "r") as f:
        agentic_predictions = json.load(f)
    print(f"Resumed {len(agentic_predictions):,} existing entries.")
else:
    agentic_predictions = {
        qid: {
            **info,
            "agent_triggered": False,
            "agent_switched_to_b": False,
            "action_note": "Fast-Path Baseline"
        }
        for qid, info in step3_baseline.items()
    }

switched_b = sum(1 for p in agentic_predictions.values() if p.get("agent_switched_to_b", False))
cat_c_refreshed = 0
loop_start = time.time()

remaining_ambiguous = [qid for qid in ambiguous_qids if not agentic_predictions[qid].get("agent_triggered", False)]
print(f"Processing {len(remaining_ambiguous)} / {len(ambiguous_qids)} remaining ambiguous queries...")

for i, qid in enumerate(tqdm(remaining_ambiguous, desc="Fine-Tuned 13B Agentic Loop")):
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

    # Select best sections for both candidates
    sec1_text, score1 = select_best_section(question, kb_dict.get(url1))
    if not sec1_text:
        sec1_text = b_info.get("top1_section", "")
    sec2_text, score2 = select_best_section(question, kb_dict.get(url2))
    if not sec2_text:
        secs2 = reconstruct_wiki_sections(kb_dict[url2]) if kb_dict.get(url2) else []
        sec2_text = secs2[0] if secs2 else ""

    # Fine-tuned 13B single-pass visual verification
    choice, dec = ask_visual_choice_finetuned(raw_img, title1, title2)
    switch_to_b = (choice == "B") or (score2 > score1 + 2.0 and score2 > 0.0)

    new_pred = b_info["prediction"]
    action_note = "Kept Candidate 1 (Baseline Evidence)"

    if switch_to_b:
        switched_b += 1
        action_note = f"Switched to {title2} (Cat B LoRA 13B: {dec})"
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

    if (i + 1) % 50 == 0 or (i + 1) == len(remaining_ambiguous):
        with open(checkpoint_path, "w") as f:
            json.dump(agentic_predictions, f, indent=2)
        elapsed = time.time() - loop_start
        print(f"  Checkpoint @ {i+1}/{len(remaining_ambiguous)} | Elapsed: {elapsed/60:.1f} min | Switches: {switched_b}")

total_loop_time = time.time() - loop_start
print(f"\nAgentic Loop Finished in {total_loop_time:.1f}s.")
print(f"Visual switches to Candidate 2: {switched_b}")
print(f"Category C section refreshes:   {cat_c_refreshed}")

# %% [markdown]
# ### Cell 11: Final Zero-Leakage Evaluation & Comparative Results Table

# %%
print("\n" + "=" * 70)
print("CELL 11: FINAL ZERO-LEAKAGE EVALUATION (FINE-TUNED 13B CONTROLLER)")
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
    if was_right:
        baseline_correct_final += 1
    if now_right:
        final_correct += 1
    if not was_right and now_right:
        recovered_events.append({
            "qid": qid,
            "question": b_info.get("question", ""),
            "action": a_info.get("action_note", ""),
            "old_pred": b_info.get("prediction", ""),
            "new_pred": a_info.get("prediction", ""),
            "gt": gt
        })
    elif was_right and not now_right:
        degraded_events.append({
            "qid": qid,
            "question": b_info.get("question", ""),
            "action": a_info.get("action_note", ""),
            "old_pred": b_info.get("prediction", ""),
            "new_pred": a_info.get("prediction", ""),
            "gt": gt
        })
    elif was_right and now_right:
        preserved_correct += 1

final_acc   = final_correct / total_q * 100.0
base_acc    = baseline_correct_final / total_q * 100.0
delta_acc   = final_acc - base_acc
net_rec     = final_correct - baseline_correct_final
avg_calls   = 1.0 + (len(ambiguous_qids) / total_q)

# Calculate McNemar Chi-squared and p-value
b_count = len(degraded_events)
c_count = len(recovered_events)
discordant = b_count + c_count
if discordant > 0:
    mcnemar_chi2 = ((abs(c_count - b_count) - 1.0) ** 2) / discordant
    win_rate = (c_count / discordant) * 100.0
else:
    mcnemar_chi2 = 0.0
    win_rate = 0.0

print(f"\n{'Metric':<35} | {'Baseline (OMGM)':<20} | {'Ours (LoRA 13B)':<20} | Delta")
print("-" * 90)
print(f"{'Overall VQA Accuracy':<35} | {base_acc:6.2f}% ({baseline_correct_final}/{total_q}) | {final_acc:6.2f}% ({final_correct}/{total_q})   | {delta_acc:+.2f}% ({net_rec:+d})")
print(f"{'Ambiguous Queries Triggered':<35} | {'N/A':<20} | {len(ambiguous_qids):<20} | {len(ambiguous_qids)/total_q*100:.1f}%")
print(f"{'Gross Recoveries (Wrong->Right)':<35} | {'0 (rigid)':<20} | {c_count:<20} | +{c_count}")
print(f"{'Degradations (Right->Wrong)':<35} | {'0 (rigid)':<20} | {b_count:<20} | -{b_count}")
print(f"{'Net Improvement':<35} | {'0':<20} | {net_rec:+d}{' queries':<19} | Honest Net")
print(f"{'Decision Win-Rate on Flips':<35} | {'--':<20} | {win_rate:6.2f}%{'':13} | ({c_count}/{discordant})")
print(f"{'McNemar Chi-Squared (p-val)':<35} | {'--':<20} | {mcnemar_chi2:6.3f}{'':14} | Chi2 stat")
print(f"{'Avg. Generator Calls/Query':<35} | {'1.00':<20} | {avg_calls:.2f}{'':18} | +{avg_calls-1:.2f} overhead")
print("=" * 90)

# %% [markdown]
# ### Cell 12: Save All Results & Output Files

# %%
print("\n" + "=" * 70)
print("CELL 12: SAVING ALL RESULTS (13B)")
print("=" * 70)

summary = {
    "model": "LLaVA-1.5-13B + LoRA Verifier (Fine-Tuned, Symmetric 6k Pairs)",
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
    "decision_win_rate_pct": round(win_rate, 2),
    "mcnemar_chi2": round(mcnemar_chi2, 4),
    "average_calls_per_query": round(avg_calls, 2),
    "loop_time_minutes": round(total_loop_time / 60, 2),
}

with open(f"{OUTPUT_DIR}/finetuned_13b_results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(f"{OUTPUT_DIR}/finetuned_13b_final_predictions.json", "w") as f:
    json.dump(agentic_predictions, f, indent=2)

with open(f"{OUTPUT_DIR}/finetuned_13b_recovered_queries.json", "w") as f:
    json.dump({"recovered_events": recovered_events, "degraded_events": degraded_events}, f, indent=2)

print(f"Saved artifacts to '{OUTPUT_DIR}/':")
for fn in os.listdir(OUTPUT_DIR):
    sz = os.path.getsize(os.path.join(OUTPUT_DIR, fn)) / 1024
    print(f"  - {fn} ({sz:.1f} KB)")
print("\nDone! Download these files from Kaggle's Output tab.")


# %% [code]
# =============================================================================
# CELL 13: CATEGORY B + C AGENTIC CONTROLLER (MULTI-SECTION EXPANSION)
# =============================================================================
print("\n" + "=" * 70)
print("CELL 13: CATEGORY B + C CONTROLLER (MULTI-SECTION CONTEXT EXPANSION)")
print("=" * 70)

def select_top2_sections_combined(question: str, entry: Optional[KnowledgeEntry], max_chars: int = 3200) -> Tuple[str, str]:
    """
    Returns (top1_section, combined_top2_sections).
    Combines Top-1 and Top-2 sections to solve Category C (Section Selection Errors).
    """
    if not entry:
        return "", ""
    sections = reconstruct_wiki_sections(entry)
    if not sections:
        return "", ""
    if len(sections) == 1:
        return sections[0], sections[0]
        
    # Rank sections using BGE
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
            
    # Fallback to first two sections
    sec1 = sections[0]
    sec2 = sections[1] if len(sections) > 1 else ""
    combined = f"{sec1}\n\n--- [Additional Section Evidence] ---\n\n{sec2}" if sec2 else sec1
    return sec1, combined[:max_chars]

# Initialize predictions dictionary for Category B+C
cat_bc_predictions = {
    qid: {
        **info,
        "agent_triggered": False,
        "agent_switched_to_b": False,
        "action_note": "Fast-Path Baseline"
    }
    for qid, info in step3_baseline.items()
}

switched_b_count = 0
cat_c_expanded_count = 0
loop_start = time.time()

print(f"Executing Category B+C Controller on {len(ambiguous_qids)} ambiguous queries...")

for i, qid in enumerate(tqdm(ambiguous_qids, desc="Category B+C Controller")):
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

total_time = time.time() - loop_start
print(f"\nCategory B+C Controller Finished in {total_time/60:.1f} minutes.")
print(f"  Switches to Candidate 2 (Cat B) : {switched_b_count}")
print(f"  Multi-Section Expansions (Cat C): {cat_c_expanded_count}")

# Save Category B+C predictions
os.makedirs(OUTPUT_DIR, exist_ok=True)
bc_pred_path = os.path.join(OUTPUT_DIR, "finetuned_13b_cat_bc_predictions.json")
with open(bc_pred_path, "w") as f:
    json.dump(cat_bc_predictions, f, indent=2)
print(f"Saved: {bc_pred_path}")


# %% [code]
# =============================================================================
# CELL 14: CATEGORY B + C ZERO-LEAKAGE SCORECARD & EVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("CELL 14: CATEGORY B + C SCORECARD")
print("=" * 70)

base_correct = 0
agent_correct = 0
recovered = []
degraded = []

for qid, b_item in step3_baseline.items():
    a_item = cat_bc_predictions[qid]
    gt = b_item.get("ground_truth_answers", "")
    
    b_match = check_match(b_item.get("prediction", ""), gt)
    a_match = check_match(a_item.get("prediction", ""), gt)
    
    if b_match:
        base_correct += 1
    if a_match:
        agent_correct += 1
        
    if not b_match and a_match:
        recovered.append(qid)
    elif b_match and not a_match:
        degraded.append(qid)

total_q = len(step3_baseline)
base_acc = base_correct / total_q * 100.0
agent_acc = agent_correct / total_q * 100.0
net_gain = len(recovered) - len(degraded)

print(f"\n{'Metric':<35} | {'Baseline (OMGM)':<20} | {'Ours (Cat B+C 13B)':<20} | {'Delta':<15}")
print("-" * 96)
print(f"{'Overall VQA Accuracy':<35} | {base_acc:.2f}% ({base_correct}/{total_q}){'':<4} | {agent_acc:.2f}% ({agent_correct}/{total_q}){'':<4} | +{agent_acc - base_acc:+.2f}% (+{net_gain})")
print(f"{'Gross Recoveries (Wrong->Right)':<35} | {'0 (rigid)':<20} | {len(recovered):<20} | +{len(recovered)}")
print(f"{'Degradations (Right->Wrong)':<35} | {'0 (rigid)':<20} | {len(degraded):<20} | -{len(degraded)}")
print(f"{'Net Improvement':<35} | {'0':<20} | {net_gain:+d} queries{'':<10} | Honest Net")
win_rate = (len(recovered) / (len(recovered) + len(degraded)) * 100) if (recovered or degraded) else 0.0
print(f"{'Decision Win-Rate on Flips':<35} | {'--':<20} | {win_rate:.2f}%{'':<14} | ({len(recovered)}/{len(recovered)+len(degraded)})")
print("=" * 96)
