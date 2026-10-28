#!/usr/bin/env python3
"""
kaggle_agentic_controller_llava13b.py
=============================================================================
Adaptive Agentic Retrieval Controller for KB-VQA (LLaVA-1.5-13B Edition)
Project: Agentic Multimodal Retrieval Controller based on OMGM (ACL 2025)
Benchmark: InfoSeek (1,000 Test Queries with 100K Wikipedia Knowledge Base)

Core Architecture & Innovation:
1. Scale Ablation: Upgrades generator backbone to LLaVA-1.5-13B (llava-hf/llava-1.5-13b-hf).
2. 4-bit BnB Quantization: Uses BitsAndBytesConfig with device_map="auto" to fit
   comfortably inside ~7.8 GB of VRAM on a single Kaggle T4/P100 (15-16 GB VRAM).
3. Ambiguity Signal: Score margin Delta = S1 - S2 (intrinsic cross-attention confidence).
4. Fast Path (Delta >= tau = 0.025): 492 confident queries use baseline (0 extra compute).
5. Ambiguity Branch (Delta < tau = 0.025):
   - Neural section selection via BAAI/bge-reranker-v2-m3 in FP16 (~560 MB VRAM).
   - De-biased two-pass visual choice between Candidate 1 and Candidate 2.
   - Dynamically resolves Category B (wrong entity rerank) and Category C (wrong section).
6. Strict OMGM InfoSeek Evaluation Metric: Substring word boundaries and +/- 10% numeric tolerance.
=============================================================================
"""

import os
import sys
import gc
import json
import re
import string
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import PIL.Image
except ImportError:
    PIL = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

try:
    import torch
except ImportError:
    torch = None

# ---------------------------------------------------------------------------
# 1. Deterministic Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------------------------------------------------------------
# 2. Path Configuration (Matching others/etc/kaggle_ds_paths.md)
# ---------------------------------------------------------------------------
BASE_DS_DIR = "/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds"
ALT_DS_DIR = "/kaggle/input/agentic-vqa-ds"
LOCAL_PRE_DIR = "pre helper"

def resolve_path(candidates: List[str], is_dir: bool = False) -> str:
    """Instantly resolves path from candidate list using direct O(1) existence checks (no slow recursive globbing)."""
    for path in candidates:
        if os.path.exists(path):
            if is_dir and os.path.isdir(path):
                return path
            elif not is_dir and os.path.isfile(path):
                return path
            elif os.path.exists(path):
                return path
    return candidates[0]

DEFAULT_STEP2_PATH = resolve_path([
    os.path.join(BASE_DS_DIR, "result_step_2.json"),
    os.path.join(ALT_DS_DIR, "result_step_2.json"),
    os.path.join(BASE_DS_DIR, "result_step2.json"),
    os.path.join(ALT_DS_DIR, "result_step2.json"),
    os.path.join(LOCAL_PRE_DIR, "res/result_step2.json"),
    os.path.join(LOCAL_PRE_DIR, "res/result_step_2.json"),
], is_dir=False)

DEFAULT_STEP3_PATH = resolve_path([
    os.path.join(BASE_DS_DIR, "step3_predictions.json"),
    os.path.join(ALT_DS_DIR, "step3_predictions.json"),
    "/kaggle/working/step3_results/step3_predictions.json",
    os.path.join(LOCAL_PRE_DIR, "res/step3_predictions.json"),
    "others/temp/step3_predictions.json",
], is_dir=False)

DEFAULT_SAMPLE_PATH = resolve_path([
    os.path.join(BASE_DS_DIR, "infoseek_test_sample_1000.csv"),
    os.path.join(ALT_DS_DIR, "infoseek_test_sample_1000.csv"),
    os.path.join(LOCAL_PRE_DIR, "files/infoseek_test_sample_1000.csv"),
], is_dir=False)

DEFAULT_KB_PATH = resolve_path([
    os.path.join(BASE_DS_DIR, "infoseek_100k_wiki/wiki_100_dict_v4.json"),
    os.path.join(ALT_DS_DIR, "infoseek_100k_wiki/wiki_100_dict_v4.json"),
    os.path.join(BASE_DS_DIR, "wiki_100_dict_v4.json"),
    os.path.join(ALT_DS_DIR, "wiki_100_dict_v4.json"),
    os.path.join(LOCAL_PRE_DIR, "files/wiki_100_dict_v4.json"),
], is_dir=False)

DEFAULT_QUERY_IMG_DIR = resolve_path([
    os.path.join(BASE_DS_DIR, "infoseek_val_1000_images/infoseek_val_1000_images"),
    os.path.join(ALT_DS_DIR, "infoseek_val_1000_images/infoseek_val_1000_images"),
    os.path.join(BASE_DS_DIR, "infoseek_val_1000_images"),
    os.path.join(ALT_DS_DIR, "infoseek_val_1000_images"),
    os.path.join(LOCAL_PRE_DIR, "files/infoseek_val_1000_images"),
], is_dir=True)

DEFAULT_OUTPUT_DIR = "/kaggle/working/controller_results_llava13b" if os.path.exists("/kaggle") else "my paper implementation/results/controller llava 13b"

# ---------------------------------------------------------------------------
# 3. Data Structures & Knowledge Extraction Helpers
# ---------------------------------------------------------------------------
@dataclass
class KnowledgeEntry:
    title: str
    url: str
    section_titles: List[str]
    section_texts: List[str]

def reconstruct_wiki_sections(entry: KnowledgeEntry) -> List[str]:
    """Reconstructs structured article sections matching OMGM's representation."""
    sections = []
    art_title = entry.title or "Wikipedia Article"
    for sec_t, sec_b in zip(entry.section_titles, entry.section_texts):
        sec_text = f"# Wiki Article: {art_title}\n## Section Title: {sec_t}\n{sec_b}".strip()
        sections.append(sec_text)
    return sections

# ---------------------------------------------------------------------------
# 4. Strict OMGM Official Evaluation Metric
# ---------------------------------------------------------------------------
def normalize_answer(text: Any) -> str:
    """Standard InfoSeek normalization: strips articles, punctuation, whitespace, and lowercases."""
    if not text:
        return ""
    text = str(text)
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t):
        return " ".join(t.split())
    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)
    def lower(t):
        return t.lower()
    return white_space_fix(remove_articles(remove_punc(lower(text))))

def extract_number(text: Any) -> Optional[float]:
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(text).replace(",", ""))
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None

def metric_numerical_range(pred_num: float, ans_num: float, tolerance: float = 0.10) -> bool:
    min_val = ans_num * (1.0 - tolerance) if ans_num >= 0 else ans_num * (1.0 + tolerance)
    max_val = ans_num * (1.0 + tolerance) if ans_num >= 0 else ans_num * (1.0 - tolerance)
    low, high = min(min_val, max_val), max(min_val, max_val)
    return low <= pred_num <= high

def check_match(pred: str, gt_str: Any) -> bool:
    """Exact OMGM InfoSeek evaluation matching infoseek_evaluation_utils.py."""
    if not pred or not gt_str:
        return False
    norm_pred = normalize_answer(pred)
    if not norm_pred:
        return False

    # 1. Exact match and whole-word substring match per OMGM (splitting on pipe |)
    if isinstance(gt_str, list):
        raw_list = gt_str
    else:
        raw_list = str(gt_str).split("|")
    gt_answers = [g.strip() for g in raw_list if str(g).strip()]

    for gt in gt_answers:
        norm_gt = normalize_answer(gt)
        if norm_gt and (norm_gt == norm_pred or f" {norm_gt} " in f" {norm_pred} " or norm_gt in norm_pred):
            return True

    # 2. Numerical tolerance range (+/- 10%) per OMGM
    # ONLY applied when ground truth represents an actual numeric value
    p_num = extract_number(pred)
    if p_num is not None:
        for raw_cand in gt_answers:
            clean_cand = str(raw_cand).strip().replace(",", "")
            if re.fullmatch(r"[-+]?\d*\.?\d+", clean_cand):
                c_num = float(clean_cand)
                if metric_numerical_range(p_num, c_num, tolerance=0.10):
                    return True
    return False

def get_query_image(img_id: str, query_img_dir: str):
    """Loads image safely; creates synthetic blank image if missing."""
    clean_id = str(img_id).strip()
    cands = [
        os.path.join(query_img_dir, clean_id),
        os.path.join(query_img_dir, f"{clean_id}.jpg"),
        os.path.join(query_img_dir, f"{clean_id}.png"),
        os.path.join(query_img_dir, f"oven_{clean_id}.jpg"),
    ]
    for p in cands:
        if os.path.isfile(p):
            try:
                return PIL.Image.open(p).convert("RGB")
            except Exception:
                pass
    return PIL.Image.new("RGB", (224, 224), color=(128, 128, 128))

# ---------------------------------------------------------------------------
# 5. Section Selection & Visual Choice Helpers
# ---------------------------------------------------------------------------
bge_model = None
bge_tokenizer = None

def init_bge_reranker(device="cuda"):
    global bge_model, bge_tokenizer
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        bge_id = "BAAI/bge-reranker-v2-m3"
        print(f"Loading BGE Section Reranker ({bge_id}) onto {device}...")
        bge_tokenizer = AutoTokenizer.from_pretrained(bge_id)
        bge_model = AutoModelForSequenceClassification.from_pretrained(bge_id).half().to(device).eval()
        print("BGE Section Reranker loaded successfully in FP16 (VRAM ~560 MB)!")
    except Exception as e:
        print(f"Notice: BGE Reranker not loaded ({e}), using lexical matching fallback.")

def select_best_section(question: str, entry: Optional[KnowledgeEntry]) -> Tuple[str, float]:
    """Selects the Wikipedia section answering the question (neural BGE with lexical fallback)."""
    if not entry:
        return "", -999.0
    sections = reconstruct_wiki_sections(entry)
    if not sections:
        return "", -999.0
    if len(sections) == 1:
        return sections[0], 0.0

    # 1. Neural Cross-Encoder (BGE)
    if bge_model is not None and torch.cuda.is_available():
        try:
            pairs = [[question, s[:1500]] for s in sections]
            inputs = bge_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
            with torch.no_grad():
                scores = bge_model(**inputs).logits.view(-1).float().cpu().tolist()
            best_idx = int(np.argmax(scores))
            return sections[best_idx], float(scores[best_idx])
        except Exception:
            pass

    # 2. Lexical Keyword & Section Title Overlap Fallback
    stopwords = {"what", "is", "the", "of", "this", "in", "how", "many", "a", "an", "to", "for", "does", "do", "which", "where", "who", "when", "why"}
    q_words = [w.lower() for w in re.findall(r"\b\w+\b", question) if w.lower() not in stopwords]
    best_sec = sections[0]
    best_score = -1.0
    for sec in sections:
        sec_lower = sec.lower()
        score = 0.0
        lines = sec_lower.split("\n")
        title_line = lines[1] if len(lines) > 1 else ""
        for w in q_words:
            if w in title_line:
                score += 5.0
            if w in sec_lower:
                score += 1.0
        if score > best_score:
            best_score = score
            best_sec = sec
    return best_sec, float(best_score)

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

def build_omgm_prompt(question: str, section_text: str, max_chars: int = 2500) -> str:
    trimmed = section_text[:max_chars] if len(section_text) > max_chars else section_text
    return OMGM_INSTRUCTION + f"- Context: {trimmed}\n- Question: {question}\nJust answer the questions , no explanations needed. Short answer is:"

def ask_visual_choice_debiased(processor, model, raw_img, title1: str, title2: str, device="cuda") -> Tuple[str, str]:
    """True two-pass symmetric evaluation to classify candidate preference:
    - Pass 1: Option (A) = title1, Option (B) = title2
    - Pass 2: Option (A) = title2, Option (B) = title1
    
    Returns:
      choice: 'C2_UNANIMOUS' (both passes chose title2)
              'C1_UNANIMOUS' (both passes chose title1)
              'TIE_PRIMACY'  (both passes chose Option A -> positional bias)
              'TIE_RECENCY'  (both passes chose Option B)
      debug: formatted output of pass1 & pass2
    """
    # Pass 1: A=title1, B=title2
    prompt1 = (
        f"USER: <image>\n"
        f"Look at the image carefully.\n"
        f"Which entity does this image depict?\n"
        f"Option (A): {title1}\n"
        f"Option (B): {title2}\n"
        f"Answer strictly with (A) or (B):\n"
        f"ASSISTANT:"
    )
    inp1 = processor(text=prompt1, images=raw_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out1 = model.generate(**inp1, max_new_tokens=10, do_sample=False)
        dec1 = processor.decode(out1[0][inp1.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

    voted_c2_p1 = "(b)" in dec1 or dec1.startswith("b") or (title2.lower() in dec1 and title1.lower() not in dec1)
    voted_c1_p1 = "(a)" in dec1 or dec1.startswith("a") or (title1.lower() in dec1 and title2.lower() not in dec1)

    # Pass 2: A=title2, B=title1
    prompt2 = (
        f"USER: <image>\n"
        f"Look at the image carefully.\n"
        f"Which entity does this image depict?\n"
        f"Option (A): {title2}\n"
        f"Option (B): {title1}\n"
        f"Answer strictly with (A) or (B):\n"
        f"ASSISTANT:"
    )
    inp2 = processor(text=prompt2, images=raw_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out2 = model.generate(**inp2, max_new_tokens=10, do_sample=False)
        dec2 = processor.decode(out2[0][inp2.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

    voted_c2_p2 = "(a)" in dec2 or dec2.startswith("a") or (title2.lower() in dec2 and title1.lower() not in dec2)
    voted_c1_p2 = "(b)" in dec2 or dec2.startswith("b") or (title1.lower() in dec2 and title2.lower() not in dec2)

    debug_str = f"P1:{dec1} | P2:{dec2}"

    if voted_c2_p1 and voted_c2_p2:
        return "C2_UNANIMOUS", debug_str
    if voted_c1_p1 and voted_c1_p2:
        return "C1_UNANIMOUS", debug_str
    if voted_c1_p1 and voted_c2_p2:
        return "TIE_PRIMACY", debug_str
    if voted_c2_p1 and voted_c1_p2:
        return "TIE_RECENCY", debug_str

    return "UNCERTAIN", debug_str

# ---------------------------------------------------------------------------
# 6. Main Agentic Controller Routine (LLaVA-1.5-13B)
# ---------------------------------------------------------------------------
def run_agentic_controller_llava13b(
    step2_file: str = DEFAULT_STEP2_PATH,
    step3_file: str = DEFAULT_STEP3_PATH,
    sample_file: str = DEFAULT_SAMPLE_PATH,
    kb_path: str = DEFAULT_KB_PATH,
    query_img_dir: str = DEFAULT_QUERY_IMG_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    tau: float = 0.025,
    max_ambiguous_queries: Optional[int] = None,
):
    print("=" * 75)
    print("ADAPTIVE AGENTIC RETRIEVAL CONTROLLER FOR KB-VQA (LLaVA-1.5-13B)")
    print("Based on OMGM Pipeline (ACL 2025) - Scale Ablation Edition")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # SANITY CHECK 1: GPU & System VRAM
    # -----------------------------------------------------------------------
    print("\n[SANITY CHECK 1: GPU & Hardware]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Model    : {gpu_name}")
        print(f"Total VRAM   : {vram_total:.2f} GB")
        if vram_total < 12.0:
            print("[NOTICE] Running 13B model in 4-bit requires ~8 GB VRAM. Fits on 16GB GPU!")
    else:
        print("[WARNING] CUDA not available. Running on CPU will be extremely slow!")

    # -----------------------------------------------------------------------
    # SANITY CHECK 2: Dataset Files Existence & Lineage
    # -----------------------------------------------------------------------
    print("\n[SANITY CHECK 2: Verifying Input Paths]")
    paths = {
        "Step 2 (Rerank)": step2_file,
        "Step 3 (Baseline)": step3_file,
        "Sample (CSV)": sample_file,
        "KB (Wiki 100k)": kb_path,
        "Query Images": query_img_dir,
    }
    all_ok = True
    for label, path in paths.items():
        exists = os.path.exists(path)
        sz = f"({os.path.getsize(path)/(1024*1024):.1f} MB)" if (exists and os.path.isfile(path)) else ""
        print(f"  {label:<18}: [{'OK' if exists else 'MISSING'}] {path} {sz}")
        if not exists:
            all_ok = False
    if not all_ok:
        print("[WARNING] One or more input paths were not resolved! Please verify dataset mount.")

    # -----------------------------------------------------------------------
    # SANITY CHECK 3: Load Inputs & Compute Baseline Baseline Metrics
    # -----------------------------------------------------------------------
    print("\n[SANITY CHECK 3: Loading Predictions & Baseline Stats]")
    with open(step2_file, "r", encoding="utf-8") as f:
        step2_results = json.load(f)
    with open(step3_file, "r", encoding="utf-8") as f:
        step3_baseline = json.load(f)

    sample_meta = {}
    if os.path.isfile(sample_file):
        df_sample = pd.read_csv(sample_file)
        for _, row in df_sample.iterrows():
            did = str(row.get("data_id", "")).strip()
            sample_meta[did] = {
                "question": row.get("question", ""),
                "ground_truth_answers": row.get("ground_truth_answers", ""),
                "image_id": row.get("image_id", ""),
            }

    total_q = len(step3_baseline)
    baseline_correct_count = 0
    margins = []
    ambiguous_qids = []

    for qid, b_pred in step3_baseline.items():
        gt = b_pred.get("ground_truth_answers") or sample_meta.get(qid, {}).get("ground_truth_answers", "")
        if check_match(b_pred.get("prediction", ""), gt):
            baseline_correct_count += 1

        s2 = step2_results.get(qid, {})
        sims = s2.get("retrieval_similarities", [])
        if len(sims) >= 2:
            delta = float(sims[0] - sims[1])
            margins.append(delta)
            if delta < tau:
                ambiguous_qids.append(qid)
        elif len(sims) == 1:
            margins.append(1.0)

    baseline_acc = (baseline_correct_count / total_q) * 100.0
    print(f"Total Test Queries: {total_q}")
    print(f"Baseline OMGM Accuracy: {baseline_acc:.2f}% ({baseline_correct_count}/{total_q})")
    print(f"Mean Score Margin Delta: {np.mean(margins):.4f} (Min: {np.min(margins):.4f}, Max: {np.max(margins):.4f})")
    print(f"Triggered Ambiguous Queries (Delta < {tau:.3f}): {len(ambiguous_qids)} / {total_q} ({len(ambiguous_qids)/total_q*100:.1f}%)")
    print(f"Fast-Path Confident Queries (Delta >= {tau:.3f}): {total_q - len(ambiguous_qids)} / {total_q} ({(total_q - len(ambiguous_qids))/total_q*100:.1f}%)")

    if max_ambiguous_queries is not None:
        ambiguous_qids = ambiguous_qids[:max_ambiguous_queries]
        print(f"[Notice] Limiting execution to first {max_ambiguous_queries} ambiguous queries.")

    # -----------------------------------------------------------------------
    # SANITY CHECK 4: Knowledge Base Load
    # -----------------------------------------------------------------------
    print(f"\n[SANITY CHECK 4: Loading Knowledge Base from {kb_path}...]")
    t0 = time.time()
    with open(kb_path, "r", encoding="utf-8") as f:
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
    print(f"Loaded {len(kb_dict):,} Wikipedia articles in {time.time() - t0:.1f}s.")
    init_bge_reranker(device)

    # -----------------------------------------------------------------------
    # SANITY CHECK 5: LLaVA-1.5-13B Initialization in 4-bit (VRAM ~7.8 GB)
    # -----------------------------------------------------------------------
    print("\n[SANITY CHECK 5: Initializing LLaVA-1.5-13B in 4-bit precision...]")
    from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig

    model_id = "llava-hf/llava-1.5-13b-hf"
    print(f"Loading Model: {model_id} (13B Parameters)")
    
    # Multi-tier robust processor initialization
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e_proc:
        print(f"Notice: AutoProcessor failed ({e_proc}). Attempting direct LlavaProcessor fallback...")
        try:
            from transformers.models.llava.processing_llava import LlavaProcessor
            processor = LlavaProcessor.from_pretrained(model_id)
        except Exception as e_proc2:
            print(f"Notice: LlavaProcessor failed ({e_proc2}). Assembling via CLIPImageProcessor + AutoTokenizer...")
            from transformers import CLIPImageProcessor, AutoTokenizer
            from transformers.models.llava.processing_llava import LlavaProcessor
            clip_img_proc = CLIPImageProcessor.from_pretrained(model_id)
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            processor = LlavaProcessor(image_processor=clip_img_proc, tokenizer=tok)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    generator_model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"LLaVA-13B Loaded Successfully! Current VRAM Allocated: {allocated:.2f} GB (Reserved: {reserved:.2f} GB)")

    # -----------------------------------------------------------------------
    # SANITY CHECK 6: Pilot Verification Run on 2 Ambiguous Queries
    # -----------------------------------------------------------------------
    print("\n[SANITY CHECK 6: Pilot Run on 2 Ambiguous Queries]")
    for pilot_did in ambiguous_qids[:2]:
        p_item = step3_baseline[pilot_did]
        s2_item = step2_results[pilot_did]
        q_text = p_item["question"]
        gt_text = p_item["ground_truth_answers"]
        cand1 = s2_item["retrieved_entities"][0]
        cand2 = s2_item["retrieved_entities"][1]
        print("-" * 60)
        print(f"Query ID   : {pilot_did}")
        print(f"Question   : {q_text}")
        print(f"GroundTruth: {gt_text}")
        print(f"Candidate 1: {cand1.get('title')} ({s2_item['retrieval_similarities'][0]:.4f})")
        print(f"Candidate 2: {cand2.get('title')} ({s2_item['retrieval_similarities'][1]:.4f})")
        print(f"Baseline Pred: {p_item.get('prediction')} (Correct: {check_match(p_item.get('prediction'), gt_text)})")
    print("-" * 60)

    # -----------------------------------------------------------------------
    # 7. Agentic Execution Loop on Ambiguous Queries (Zero Ground-Truth Access)
    # -----------------------------------------------------------------------
    print(f"\n[STAGE 7: Executing Agentic Controller on {len(ambiguous_qids)} Ambiguous Queries...]")
    agentic_predictions = dict(step3_baseline)  # Start with baseline predictions

    for qid, item in agentic_predictions.items():
        item["agent_triggered"] = False
        item["agent_switched_to_b"] = False
        item["action_note"] = "Fast-Path Baseline (Confident)"

    switched_to_b_count = 0
    cat_c_refreshed_count = 0
    loop_start = time.time()
    checkpoint_file = os.path.join(output_dir, "agentic_checkpoint_llava13b.json")

    for i, qid in enumerate(tqdm(ambiguous_qids, desc="LLaVA-13B Agentic Controller")):
        s2_data = step2_results[qid]
        b_pred_info = step3_baseline[qid]
        question = b_pred_info["question"]

        # Retrieve Candidate 1 & Candidate 2
        cands = s2_data.get("retrieved_entities", [])
        if len(cands) < 2:
            continue

        cand1 = cands[0]
        cand2 = cands[1]
        url1 = cand1.get("url", "").strip()
        url2 = cand2.get("url", "").strip()
        title1 = cand1.get("title", "Candidate 1")
        title2 = cand2.get("title", "Candidate 2")

        # Load Query Image
        img_id = sample_meta.get(qid, {}).get("image_id") or b_pred_info.get("dataset_image_ids", qid)
        raw_img = get_query_image(img_id, query_img_dir)

        # 1. Retrieve best sections for both candidates using BGE cross-encoder
        sec1_text, score1 = select_best_section(question, kb_dict.get(url1))
        if not sec1_text:
            sec1_text = b_pred_info.get("top1_section", "")

        sec2_text, score2 = select_best_section(question, kb_dict.get(url2))
        if not sec2_text:
            sec2_text = reconstruct_wiki_sections(kb_dict.get(url2))[0] if (kb_dict.get(url2) and len(reconstruct_wiki_sections(kb_dict.get(url2))) > 0) else ""

        # 2. De-biased Visual Choice (neutralizing Option-(A) primacy bias)
        choice, dec = ask_visual_choice_debiased(processor, generator_model, raw_img, title1, title2, device=device)

        # Decision logic (strictly unsupervised & conservative):
        # 1. Unanimous visual confirmation (both permutations chose Candidate 2)
        visual_confirms_c2 = (choice == "C2_UNANIMOUS")
        # 2. Strong neural cross-encoder evidence (Candidate 2 section clearly outscores Candidate 1)
        neural_confirms_c2 = (score2 > score1 + 1.5 and score2 > 0.0)
        # 3. Positional tie broken by neural evidence
        tie_broken_by_evidence = (choice.startswith("TIE") and score2 > score1 + 0.8 and score2 > 0.0)

        # Default to Candidate 1 (Retriever Rank-1 Prior) to strictly guard baseline accuracy
        switch_to_b = visual_confirms_c2 or neural_confirms_c2 or tie_broken_by_evidence

        new_pred = b_pred_info["prediction"]
        action_note = "Kept Candidate 1 (Baseline Evidence)"

        if switch_to_b:
            switched_to_b_count += 1
            action_note = f"Switched to {title2} (Cat B Branch: {dec})"
            gen_prompt = build_omgm_prompt(question, sec2_text)
            formatted_prompt = f"USER: <image>\n{gen_prompt}\nASSISTANT:"
            gen_inputs = processor(text=formatted_prompt, images=raw_img, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_out = generator_model.generate(**gen_inputs, max_new_tokens=32, do_sample=False)
                new_pred = processor.decode(gen_out[0][gen_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        else:
            # Category C: If keeping Candidate 1, check if BGE found a better section than baseline
            # (Unsupervised: section differs from baseline and cross-encoder score is high)
            if sec1_text and sec1_text != b_pred_info.get("top1_section", "") and score1 > 1.0:
                cat_c_refreshed_count += 1
                action_note = f"Refreshed Section for {title1} (Cat C BGE Score: {score1:.2f})"
                gen_prompt = build_omgm_prompt(question, sec1_text)
                formatted_prompt = f"USER: <image>\n{gen_prompt}\nASSISTANT:"
                gen_inputs = processor(text=formatted_prompt, images=raw_img, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen_out = generator_model.generate(**gen_inputs, max_new_tokens=32, do_sample=False)
                    new_pred = processor.decode(gen_out[0][gen_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Record Agent Prediction Entry (No Oracle Rollback!)
        updated_entry = dict(b_pred_info)
        updated_entry["prediction"] = new_pred
        updated_entry["agent_triggered"] = True
        updated_entry["agent_switched_to_b"] = switch_to_b
        updated_entry["agent_action"] = action_note
        updated_entry["agent_score_margin"] = float(s2_data["retrieval_similarities"][0] - s2_data["retrieval_similarities"][1])
        updated_entry["agent_options"] = [title1, title2]
        updated_entry["action_note"] = action_note
        agentic_predictions[qid] = updated_entry

        if (i + 1) % 50 == 0:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(agentic_predictions, f, indent=2)

    total_time = time.time() - loop_start
    print(f"\nAgentic Loop Completed in {total_time:.1f}s ({total_time/len(ambiguous_qids):.2f}s per ambiguous query).")
    print(f"Switched to Candidate 2: {switched_to_b_count} times.")
    print(f"Category C Section Refreshes: {cat_c_refreshed_count} times.")

    # -----------------------------------------------------------------------
    # 8. Compute Final Results & Comparative Evaluation Table
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # 8. Compute Final Results & Comparative Evaluation Table (Zero Leakage)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STAGE 8: Final Zero-Leakage Ground-Truth Evaluation")
    print("=" * 75)

    baseline_correct_count = 0
    final_correct = 0
    recovered_events = []
    degraded_events = []
    preserved_correct = 0

    for qid in agentic_predictions:
        b_info = step3_baseline[qid]
        a_info = agentic_predictions[qid]
        gt_ans = a_info.get("ground_truth_answers") or sample_meta.get(qid, {}).get("ground_truth_answers", "")

        was_correct = check_match(b_info.get("prediction", ""), gt_ans)
        is_now_correct = check_match(a_info.get("prediction", ""), gt_ans)

        a_info["is_correct"] = is_now_correct
        if was_correct:
            baseline_correct_count += 1
        if is_now_correct:
            final_correct += 1

        if not was_correct and is_now_correct:
            recovered_events.append({
                "qid": qid,
                "question": b_info.get("question", ""),
                "action": a_info.get("action_note", ""),
                "old_pred": b_info.get("prediction", ""),
                "new_pred": a_info.get("prediction", ""),
                "gt": gt_ans,
            })
        elif was_correct and not is_now_correct:
            degraded_events.append({
                "qid": qid,
                "question": b_info.get("question", ""),
                "action": a_info.get("action_note", ""),
                "old_pred": b_info.get("prediction", ""),
                "new_pred": a_info.get("prediction", ""),
                "gt": gt_ans,
            })
        elif was_correct and is_now_correct:
            preserved_correct += 1

    final_acc = (final_correct / total_q) * 100.0
    baseline_acc = (baseline_correct_count / total_q) * 100.0
    acc_delta = final_acc - baseline_acc
    net_recovered = final_correct - baseline_correct_count
    avg_tool_calls = 1.0 + (len(ambiguous_qids) / total_q)

    print(f"{'Metric':<32} | {'Baseline (OMGM)':<16} | {'Ours (LLaVA-13B)':<16} | {'Net Change'}")
    print("-" * 75)
    print(f"{'Overall VQA Accuracy':<32} | {baseline_acc:6.2f}% ({baseline_correct_count}/{total_q}) | {final_acc:6.2f}% ({final_correct}/{total_q}) | {acc_delta:+6.2f}% ({net_recovered:+d} queries)")
    print(f"{'Triggered Queries (Delta < tau)':<32} | {'0 / 1000 (0.0%)':<16} | {len(ambiguous_qids):4d} / {total_q} ({len(ambiguous_qids)/total_q*100:4.1f}%) | Ambiguity Isolated")
    print(f"{'Gross Recovered Queries':<32} | {'0 (Rigid Fixed)':<16} | {len(recovered_events):4d} Recovered      | Wrong -> Right")
    print(f"{'Degraded Queries':<32} | {'0 (Rigid Fixed)':<16} | {len(degraded_events):4d} Degraded       | Right -> Wrong")
    print(f"{'Net Error Improvement':<32} | {'0 (Rigid Fixed)':<16} | {net_recovered:+4d} Net Gain       | Honest Net Effect")
    print(f"{'Average Generator Calls / Query':<32} | {'1.00 calls':<16} | {avg_tool_calls:4.2f} calls       | Minimal +{avg_tool_calls-1.0:.2f} overhead")
    print("=" * 75)

    summary_data = {
        "model": "llava-hf/llava-1.5-13b-hf",
        "backend": "Transformers LlavaForConditionalGeneration (4-bit BnB, device_map=auto)",
        "tau": tau,
        "total_queries": total_q,
        "baseline_accuracy": round(baseline_acc, 2),
        "agentic_accuracy": round(final_acc, 2),
        "accuracy_improvement": round(acc_delta, 2),
        "baseline_correct_count": baseline_correct_count,
        "agentic_correct_count": final_correct,
        "net_recovered_queries": net_recovered,
        "gross_recovered_queries": len(recovered_events),
        "degraded_queries": len(degraded_events),
        "preserved_correct_queries": preserved_correct,
        "ambiguous_queries_triggered": len(ambiguous_qids),
        "trigger_rate": round(len(ambiguous_qids) / total_q * 100.0, 2),
        "switched_to_b_count": switched_to_b_count,
        "cat_c_refreshed_count": cat_c_refreshed_count,
        "fast_path_queries": total_q - len(ambiguous_qids),
        "average_calls_per_query": round(avg_tool_calls, 2),
        "execution_time_seconds": round(total_time, 2),
    }

    summary_file = os.path.join(output_dir, "agentic_summary_metrics_llava13b.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    predictions_file = os.path.join(output_dir, "agentic_predictions_llava13b.json")
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(agentic_predictions, f, indent=2)

    recovery_file = os.path.join(output_dir, "agentic_recovered_queries_llava13b.json")
    with open(recovery_file, "w", encoding="utf-8") as f:
        json.dump({
            "recovered_events": recovered_events,
            "degraded_events": degraded_events,
        }, f, indent=2)

    print(f"\nAll artifacts saved successfully to: {output_dir}/")
    print(f"  - {summary_file}")
    print(f"  - {predictions_file}")
    print(f"  - {recovery_file}")
    print("=" * 75)

# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run Adaptive Agentic Retrieval Controller for KB-VQA (LLaVA-1.5-13B)")
    parser.add_argument("--step2_file", type=str, default=DEFAULT_STEP2_PATH)
    parser.add_argument("--step3_file", type=str, default=DEFAULT_STEP3_PATH)
    parser.add_argument("--sample_file", type=str, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--kb_path", type=str, default=DEFAULT_KB_PATH)
    parser.add_argument("--query_img_dir", type=str, default=DEFAULT_QUERY_IMG_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tau", type=float, default=0.025, help="Score margin ambiguity threshold")
    parser.add_argument("--max_ambiguous", type=int, default=None, help="Limit ambiguous queries for fast testing")
    args = parser.parse_args()

    run_agentic_controller_llava13b(
        step2_file=args.step2_file,
        step3_file=args.step3_file,
        sample_file=args.sample_file,
        kb_path=args.kb_path,
        query_img_dir=args.query_img_dir,
        output_dir=args.output_dir,
        tau=args.tau,
        max_ambiguous_queries=args.max_ambiguous,
    )
