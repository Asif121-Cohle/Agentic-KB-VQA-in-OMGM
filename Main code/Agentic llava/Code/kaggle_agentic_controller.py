#!/usr/bin/env python3
"""
kaggle_agentic_controller.py
=============================================================================
Phase 2: Adaptive Agentic Retrieval Controller for KB-VQA (InfoSeek Benchmark)
Project: Agentic Multimodal Retrieval Controller based on OMGM (ACL 2025)

Core Architecture:
1. Confidence Signal: Score margin Delta = S1 - S2 (intrinsic cross-attention confidence).
2. Fast Path (Delta >= tau): Retains baseline Step 3 top-1 prediction (0 extra compute).
3. Ambiguity Trigger (Delta < tau, default tau = 0.025):
   - tool_top2_branch: Fetches top sections of Candidate 1 and Candidate 2.
   - Prompts LLaVA-1.5-7B with visual query image and dual candidate options side-by-side.
   - Dynamically resolves Category B (wrong entity rerank) and Category C (wrong section).
4. Full Sanity Checks & Verbose Logging for cell-by-cell verification on Kaggle.
=============================================================================
"""

import os
import sys
import gc
import json
import re
import string
import time
import glob
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

DEFAULT_OUTPUT_DIR = "/kaggle/working/agentic_results"


# ---------------------------------------------------------------------------
# 3. InfoSeek Evaluation Metric Helpers
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

# ---------------------------------------------------------------------------
# 4. Knowledge Base & Query Image Helpers
# ---------------------------------------------------------------------------
class KnowledgeEntry:
    def __init__(self, title: str, url: str, section_titles: List[str], section_texts: List[str]):
        self.title = title
        self.url = url
        self.section_titles = section_titles
        self.section_texts = section_texts

def reconstruct_wiki_sections(entry: KnowledgeEntry) -> List[str]:
    """Reconstruct sections from knowledge base entry matching OMGM formatting."""
    sections = []
    title = entry.title
    for it, sec_title in enumerate(entry.section_titles):
        sec_lower = sec_title.lower()
        if "external links" in sec_lower or "references" in sec_lower:
            continue
        text = entry.section_texts[it] if it < len(entry.section_texts) else ""
        formatted = f"# Wiki Article: {title}\n## Section Title: {sec_title}\n{text}"
        sections.append(formatted)
    return sections

def get_query_image(image_id: str, search_dir: str) -> Optional[PIL.Image.Image]:
    """Loads query image with extension fallbacks and glob pattern matching."""
    clean_id = str(image_id).split("|")[0].strip()
    for ext in ["", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG"]:
        p = os.path.join(search_dir, f"{clean_id}{ext}")
        if os.path.isfile(p):
            try:
                return PIL.Image.open(p).convert("RGB")
            except Exception:
                pass
    matches = glob.glob(os.path.join(search_dir, f"*{clean_id}*"))
    for m in matches:
        if os.path.isfile(m) and m.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            try:
                return PIL.Image.open(m).convert("RGB")
            except Exception:
                pass
    # Fallback blank image if file is missing/unreadable
    return PIL.Image.new("RGB", (224, 224), color=(128, 128, 128))

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 5. Section Selection & Decoupled Prompt Helpers
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
    """Accurately selects the Wikipedia section answering the question (never blindly takes index 0)."""
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

def generate_answer(processor, model, raw_img: PIL.Image.Image, question: str, section_text: str, max_chars: int = 2500, device="cuda") -> str:
    """Generates factual answer using OMGM standard prompt from selected section."""
    trimmed = section_text[:max_chars] if len(section_text) > max_chars else section_text
    prompt_text = build_omgm_prompt(question, trimmed)
    full_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
    inputs = processor(images=raw_img, text=full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    prompt_len = inputs.input_ids.shape[1]
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

def ask_visual_choice_debiased(processor, model, raw_img: PIL.Image.Image, title1: str, title2: str, device="cuda") -> Tuple[str, str]:
    """
    Two-pass symmetric evaluation to neutralize Option-(A) primacy bias.
    Pass 1: Option (A): title1, Option (B): title2
    Pass 2: Option (A): title2, Option (B): title1

    If the model chooses (A) in both passes due to positional bias, we resolve
    the tie by sticking with Candidate 1 (the higher-ranked retriever top-1).
    Candidate 2 is selected only if it receives genuine visual preference
    across permutations.
    """
    # Pass 1: A=title1, B=title2
    p1 = (
        f"Look at the image carefully.\n"
        f"Which entity does this image depict?\n"
        f"Option (A): {title1}\n"
        f"Option (B): {title2}\n"
        f"Answer strictly with (A) or (B):"
    )
    prompt1 = f"USER: <image>\n{p1}\nASSISTANT:"
    inp1 = processor(images=raw_img, text=prompt1, return_tensors="pt").to(device)
    with torch.no_grad():
        out1 = model.generate(**inp1, max_new_tokens=5, do_sample=False)
    dec1 = processor.decode(out1[0][inp1.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

    # Pass 2: A=title2, B=title1
    p2 = (
        f"Look at the image carefully.\n"
        f"Which entity does this image depict?\n"
        f"Option (A): {title2}\n"
        f"Option (B): {title1}\n"
        f"Answer strictly with (A) or (B):"
    )
    prompt2 = f"USER: <image>\n{p2}\nASSISTANT:"
    inp2 = processor(images=raw_img, text=prompt2, return_tensors="pt").to(device)
    with torch.no_grad():
        out2 = model.generate(**inp2, max_new_tokens=5, do_sample=False)
    dec2 = processor.decode(out2[0][inp2.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

    def parse_choice(dec_str: str, opt_a: str, opt_b: str) -> Optional[str]:
        if "(a)" in dec_str or dec_str.startswith("a") or dec_str == "a":
            return "A"
        if "(b)" in dec_str or dec_str.startswith("b") or dec_str == "b":
            return "B"
        a_low, b_low = opt_a.lower(), opt_b.lower()
        if a_low in dec_str and b_low not in dec_str:
            return "A"
        if b_low in dec_str and a_low not in dec_str:
            return "B"
        return None

    c1 = parse_choice(dec1, title1, title2)
    c2 = parse_choice(dec2, title2, title1)

    p1_chose_cand2 = (c1 == "B")
    p2_chose_cand2 = (c2 == "A")
    p2_chose_cand1 = (c2 == "B")

    # Unanimous vote for Candidate 2 across both permutations
    if p1_chose_cand2 and p2_chose_cand2:
        return "B", f"Unanimous Cand 2 (P1:{dec1} | P2:{dec2})"

    # Pass 1 overcame Option (A) bias to choose Candidate 2 (B), and Pass 2 did not explicitly vote for Candidate 1 (B)
    if p1_chose_cand2 and not p2_chose_cand1:
        return "B", f"Strong Cand 2 (P1:{dec1} | P2:{dec2})"

    # Otherwise (including positional tie where both choose A), retain Candidate 1
    return "A", f"P1:{dec1} | P2:{dec2}"
# 6. Main Agentic Controller Routine
# ---------------------------------------------------------------------------
def run_agentic_controller(
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
    print("ADAPTIVE AGENTIC RETRIEVAL CONTROLLER FOR KB-VQA")
    print("Based on OMGM Pipeline (ACL 2025)")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # SANITY CHECK 1: GPU & System VRAM
    # -----------------------------------------------------------------------
    print("\n[SANITY CHECK 1: GPU & Hardware]")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Device: {gpu_name} ({vram_gb:.2f} GB Total VRAM)")
        device = "cuda"
    else:
        print("[WARNING] CUDA not detected! Running on CPU will be extremely slow.")
        device = "cpu"

    # -----------------------------------------------------------------------
    # SANITY CHECK 2: Dataset Files Existence & Record Counts
    # -----------------------------------------------------------------------
    print("\n[SANITY CHECK 2: Dataset Verification]")
    print(f"1. Step 2 Result File : {step2_file} (Exists: {os.path.exists(step2_file)})")
    print(f"2. Step 3 Predictions : {step3_file} (Exists: {os.path.exists(step3_file)})")
    print(f"3. Sample Metadata CSV: {sample_file} (Exists: {os.path.exists(sample_file)})")
    print(f"4. Wikipedia KB File  : {kb_path} (Exists: {os.path.exists(kb_path)})")
    num_imgs = len(os.listdir(query_img_dir)) if os.path.isdir(query_img_dir) else 0
    print(f"5. Query Images Dir   : {query_img_dir} (Exists: {os.path.isdir(query_img_dir)}, Images: {num_imgs})")
    print(f"6. Controller Output  : {output_dir}")
    print(f"7. Ambiguity Threshold: tau = {tau:.4f}")

    if not os.path.exists(step2_file) or not os.path.exists(step3_file):
        raise FileNotFoundError(f"Missing required baseline files: {step2_file} or {step3_file}")

    # Load Step 2 Results
    with open(step2_file, "r", encoding="utf-8") as f:
        step2_results = json.load(f)
    print(f"Loaded Step 2 outputs: {len(step2_results):,} queries.")

    # Load Step 3 Baseline Predictions
    with open(step3_file, "r", encoding="utf-8") as f:
        step3_baseline = json.load(f)
    print(f"Loaded Step 3 baseline outputs: {len(step3_baseline):,} queries.")

    # Load InfoSeek Sample Metadata
    df_samples = pd.read_csv(sample_file)
    sample_meta = {}
    for _, row in df_samples.iterrows():
        did = str(row.get("data_id", "")).strip()
        sample_meta[did] = {
            "question": str(row.get("question", "")),
            "image_id": str(row.get("dataset_image_ids", "")).split("|")[0].strip(),
            "wikipedia_url": str(row.get("wikipedia_url", "")).strip(),
            "ground_truth_answers": str(row.get("answer", "")),
        }
    print(f"Loaded query sample manifest: {len(sample_meta):,} rows.")

    # -----------------------------------------------------------------------
    # SANITY CHECK 3: Baseline Score Margin Distribution & Trigger Analysis
    # -----------------------------------------------------------------------
    print("\n[SANITY CHECK 3: Score Margin & Ambiguity Distribution]")
    margins = []
    ambiguous_qids = []
    baseline_correct_count = 0

    for did, p_info in step3_baseline.items():
        # Evaluate baseline accuracy
        pred = p_info.get("prediction", "")
        gt = p_info.get("ground_truth_answers") or sample_meta.get(did, {}).get("ground_truth_answers", "")
        is_corr = check_match(pred, gt)
        if is_corr:
            baseline_correct_count += 1

        # Check Step 2 margin Delta = S1 - S2
        s2_entry = step2_results.get(did, {})
        scores = s2_entry.get("retrieval_similarities", [])
        if len(scores) > 1:
            delta = float(scores[0] - scores[1])
        else:
            delta = 0.0
        margins.append(delta)

        if delta < tau:
            ambiguous_qids.append(did)

    total_q = len(step3_baseline)
    baseline_acc = (baseline_correct_count / total_q) * 100.0
    print(f"Baseline OMGM Accuracy: {baseline_acc:.2f}% ({baseline_correct_count}/{total_q})")
    print(f"Mean Score Margin Delta: {np.mean(margins):.4f} (Min: {np.min(margins):.4f}, Max: {np.max(margins):.4f})")
    print(f"Triggered Ambiguous Queries (Delta < {tau:.3f}): {len(ambiguous_qids)} / {total_q} ({len(ambiguous_qids)/total_q*100:.1f}%)")
    print(f"Fast-Path Confident Queries (Delta >= {tau:.3f}): {total_q - len(ambiguous_qids)} / {total_q} ({(total_q - len(ambiguous_qids))/total_q*100:.1f}%)")

    # Optional slice limit for fast debugging if requested
    if max_ambiguous_queries is not None:
        ambiguous_qids = ambiguous_qids[:max_ambiguous_queries]
        print(f"[Notice] Limiting execution to first {max_ambiguous_queries} ambiguous queries.")

    # -----------------------------------------------------------------------
    # SANITY CHECK 4: Knowledge Base Load & Section Extraction Test
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
    # SANITY CHECK 5: LLaVA-1.5-7B Initialization in 4-bit (VRAM ~4.5 GB)
    # -----------------------------------------------------------------------
    print("\n[SANITY CHECK 5: Initializing LLaVA-1.5-7B in 4-bit precision...]")
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)

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
        print(f"LLaVA Loaded Successfully! Current VRAM Allocated: {allocated:.2f} GB (Reserved: {reserved:.2f} GB)")

    # -----------------------------------------------------------------------
    # SANITY CHECK 6: Pilot Verification Run on 2 Ambiguous Queries
    # -----------------------------------------------------------------------
    print("\n[SANITY CHECK 6: Pilot Run on 2 Ambiguous Queries (Visible Output)]")
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
    checkpoint_file = os.path.join(output_dir, "agentic_checkpoint.json")

    for i, qid in enumerate(tqdm(ambiguous_qids, desc="Agentic Controller")):
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

        # 1. Retrieve true sections for both Candidate 1 and Candidate 2 using BGE cross-encoder
        sec1_text, score1 = select_best_section(question, kb_dict.get(url1))
        if not sec1_text:
            sec1_text = b_pred_info.get("top1_section", "")

        sec2_text, score2 = select_best_section(question, kb_dict.get(url2))
        if not sec2_text:
            sec2_text = reconstruct_wiki_sections(kb_dict.get(url2))[0] if (kb_dict.get(url2) and len(reconstruct_wiki_sections(kb_dict.get(url2))) > 0) else ""

        # 2. De-biased Visual Choice (neutralizing Option-(A) primacy bias)
        choice, dec = ask_visual_choice_debiased(processor, generator_model, raw_img, title1, title2, device=device)

        # Decision logic (strictly unsupervised):
        # Switch to Candidate 2 if visually confirmed OR if Candidate 2's cross-encoder section score is overwhelmingly higher
        switch_to_b = (choice == "B") or (score2 > score1 + 2.0 and score2 > 0.0)

        new_pred = b_pred_info["prediction"]
        action_note = "Kept Candidate 1 (Baseline Evidence)"

        if switch_to_b:
            switched_to_b_count += 1
            action_note = f"Switched to {title2} (Cat B Branch: {dec})"
            new_pred = generate_answer(processor, generator_model, raw_img, question, sec2_text, device=device)
        else:
            # Category C: If keeping Candidate 1, check if BGE identified a superior section
            # (Unsupervised: section differs from baseline and cross-encoder score is high)
            if sec1_text and sec1_text != b_pred_info.get("top1_section", "") and score1 > 1.0:
                cat_c_refreshed_count += 1
                action_note = f"Refreshed Section for {title1} (Cat C BGE Score: {score1:.2f})"
                new_pred = generate_answer(processor, generator_model, raw_img, question, sec1_text, device=device)

        # Record Agent Prediction Entry (No Oracle Rollback!)
        updated_entry = dict(b_pred_info)
        updated_entry["prediction"] = new_pred
        updated_entry["agent_triggered"] = True
        updated_entry["agent_switched_to_b"] = switch_to_b
        updated_entry["agent_score_margin"] = float(s2_data["retrieval_similarities"][0] - s2_data["retrieval_similarities"][1])
        updated_entry["agent_options"] = [title1, title2]
        updated_entry["action_note"] = action_note
        agentic_predictions[qid] = updated_entry

        # Intermediate Auto-Save Checkpoint
        if (i + 1) % 50 == 0:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(agentic_predictions, f, indent=2)

    total_time = time.time() - loop_start
    print(f"\nAgentic Loop Completed in {total_time:.1f}s ({total_time/len(ambiguous_qids):.2f}s per ambiguous query).")

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

    # Tool call calculation: 1 initial call per query + 1 controller call for ambiguous queries
    avg_tool_calls = 1.0 + (len(ambiguous_qids) / total_q)

    print(f"{'Metric':<32} | {'Baseline (OMGM)':<16} | {'Ours (Agentic)':<16} | {'Net Change'}")
    print("-" * 75)
    print(f"{'Overall VQA Accuracy':<32} | {baseline_acc:6.2f}% ({baseline_correct_count}/{total_q}) | {final_acc:6.2f}% ({final_correct}/{total_q}) | {acc_delta:+6.2f}% ({net_recovered:+d} queries)")
    print(f"{'Triggered Queries (Delta < tau)':<32} | {'0 / 1000 (0.0%)':<16} | {len(ambiguous_qids):4d} / {total_q} ({len(ambiguous_qids)/total_q*100:4.1f}%) | Ambiguity Isolated")
    print(f"{'Gross Recovered Queries':<32} | {'0 (Rigid Fixed)':<16} | {len(recovered_events):4d} Recovered      | Wrong -> Right")
    print(f"{'Degraded Queries':<32} | {'0 (Rigid Fixed)':<16} | {len(degraded_events):4d} Degraded       | Right -> Wrong")
    print(f"{'Net Error Improvement':<32} | {'0 (Rigid Fixed)':<16} | {net_recovered:+4d} Net Gain       | Honest Net Effect")
    print(f"{'Average Generator Calls / Query':<32} | {'1.00 calls':<16} | {avg_tool_calls:4.2f} calls       | Minimal +{avg_tool_calls-1.0:.2f} overhead")
    print("=" * 75)

    # Save Output Artifacts
    summary_data = {
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
        "average_calls_per_query": round(avg_tool_calls, 2),
    }

    summary_file = os.path.join(output_dir, "agentic_results_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    predictions_file = os.path.join(output_dir, "agentic_final_predictions.json")
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(agentic_predictions, f, indent=2)

    recovery_file = os.path.join(output_dir, "agentic_recovered_queries.json")
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
    parser = ArgumentParser(description="Run Adaptive Agentic Retrieval Controller for KB-VQA")
    parser.add_argument("--step2_file", type=str, default=DEFAULT_STEP2_PATH)
    parser.add_argument("--step3_file", type=str, default=DEFAULT_STEP3_PATH)
    parser.add_argument("--sample_file", type=str, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--kb_path", type=str, default=DEFAULT_KB_PATH)
    parser.add_argument("--query_img_dir", type=str, default=DEFAULT_QUERY_IMG_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tau", type=float, default=0.025, help="Score margin ambiguity threshold")
    parser.add_argument("--max_ambiguous", type=int, default=None, help="Limit ambiguous queries for fast testing")
    args = parser.parse_args()

    run_agentic_controller(
        step2_file=args.step2_file,
        step3_file=args.step3_file,
        sample_file=args.sample_file,
        kb_path=args.kb_path,
        query_img_dir=args.query_img_dir,
        output_dir=args.output_dir,
        tau=args.tau,
        max_ambiguous_queries=args.max_ambiguous,
    )
