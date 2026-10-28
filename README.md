# Adaptive Agentic Retrieval: Breaking the Rigid Pipeline Bottleneck in Knowledge-Based Visual Question Answering

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-red.svg)](my%20paper%20track/IEEE_Conference_Template/conference_101719.pdf)
[![Benchmark](https://img.shields.io/badge/Benchmark-InfoSeek%20(100k%20Wikipedia)-blue.svg)](https://infoseek-eval.github.io/)
[![Baseline](https://img.shields.io/badge/Baseline-OMGM%20(ACL%202025%20Main)-orange.svg)](https://arxiv.org/abs/2505.07879)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-informational.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Compute-Consumer%20GPU%20(T4%20%2F%20P100%20%2F%20RTX%203090)-purple.svg)](#hardware-requirements--engineering-optimizations)

This repository contains the official implementation, experimental artifacts, and paper sources for **"Adaptive Agentic Retrieval: Breaking the Rigid Pipeline Bottleneck in Knowledge-Based Visual Question Answering"** by Shahad Abir and Abdullah Mohammed Asif (*Department of Computer Science and Engineering, Chittagong University of Engineering and Technology*).

---

## 📌 Executive Summary & Abstract

Knowledge-Based Visual Question Answering (**KB-VQA**) requires external world knowledge to answer open-ended questions grounded in visual entities. While state-of-the-art multi-stage retrieval pipelines (such as **OMGM**, ACL 2025) search massive encyclopedic corpora, they suffer from a fundamental architectural flaw: **pipeline rigidity**. Queries traverse fixed sequential stages—coarse search, multimodal reranking, and generation—without uncertainty detection, confidence calibration, or adaptive branching.

Through an extensive empirical audit of 1,000 queries on **InfoSeek** evaluated against the **full, unpruned 100,000-article Wikipedia knowledge base**, we uncover:
1. **The Rigid Ranking Bottleneck:** **40.8% of all pipeline failures (408 queries)** stem not from retrieval absence, but from premature filtering of valid candidates already present in the candidate pool (**Category B:** 20.5% entity rerank errors; **Category C:** 20.3% section selection errors).
2. **The $3.3\times$ Score Margin Signal:** Ambiguous routing instances exhibit a sharp $3.3\times$ collapse in multimodal reranking score margin ($\Delta = S_1 - S_2 = 0.0512 \to 0.0155$), establishing a zero-cost mathematical threshold ($\tau = 0.025$) to gate agentic branching.
3. **The Zero-Shot Verification Dilemma:** Off-the-shelf zero-shot vision-language models (VLMs) fail as visual adjudicators due to severe positional primacy bias (17.5%) and instruction drift (47.5%), operating as an uncalibrated coin-flip ($\sim 46\%$ decision win-rate).
4. **Symmetric PEFT Tool Calibration:** By introducing symmetric parameter-efficient fine-tuning (LoRA $r=16$) on balanced entity pairs, positional bias is suppressed to **1.33%** while pairwise adjudication accuracy surges to **90.33%**.
5. **The $2 \times 2$ Capacity & Verification Double Dissociation:** Lightweight 7B models establish the Pareto-optimal frontier for pairwise entity verification (**+1.10% net gain, 62.8% win-rate, 44% lower latency**), whereas 13B models are structurally essential (**+1.40% net gain, 44 gross recoveries**) to resist distractor dilution during multi-section context expansion.
6. **State-of-the-Art Benchmark:** Our champion model achieves **36.30% VQA accuracy (+1.40% over OMGM baseline)** under **strict zero ground-truth leakage**, rescuing 44 failed queries.

---

## 🏗️ System Architecture

Adaptive Agentic Retrieval dynamically determines routing based on the intrinsic multimodal score margin $\Delta = S_1 - S_2$:

```
                      Query: (Image I_q, Question Q)
                                    │
                                    ▼
                     [Step 1: Coarse Entity Search]
                       CLIP ViT-B/32 + FAISS Index
                        (Top-20 Wikipedia Entities)
                                    │
                                    ▼
                   [Step 2: Multimodal IT2IT Reranking]
                     Q-Former Cross-Attention Fusion
                       Top Candidates: e_(1), e_(2)
                                    │
                         Score Margin Calculation
                            Δ = S_1 - S_2
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
       [Fast-Path: Confident Query]   [Ambiguity Zone: Adaptive Controller]
              (Δ ≥ τ = 0.025)                     (Δ < τ = 0.025)
              (49.2% of queries)                 (50.8% of queries)
                     │                             │
                     │                     Pairwise Visual Verifier
                     │                  (Symmetric LoRA-Calibrated VLM)
                     │                             │
                     │                 ┌───────────┴───────────┐
                     │                 ▼                       ▼
                     │          Verifier selects e_(2)  Verifier selects e_(1)
                     │          [Category B Recovery]   [Category C Recovery]
                     │          Switch to Candidate 2   Top-1 & Top-2 Sections
                     │                 │                Expanded Context
                     │                 └───────────┬───────────┘
                     │                             │
                     ▼                             ▼
        [Step 3: Section Selection]  [Decoupled Dual-Mode Generation]
           BGE Reranker v2-m3         LoRA Disabled (Pure Base Model Parity)
                     │                             │
                     └──────────────┬──────────────┘
                                    ▼
                         Final Grounded VQA Answer
```

### Key Components:
- **Zero-Cost Threshold Policy ($\pi(q)$):** Confident queries ($\Delta \ge 0.025$) proceed along the Fast-Path with **0 extra compute overhead**. Only uncertain queries ($\Delta < 0.025$) enter agentic verification.
- **Decoupled Dual-Mode Inference:** The LoRA adapter is activated solely during pairwise visual adjudication (`ask_visual_choice_finetuned`). When generating final answers (`generate_answer`), the adapter is disabled (`disable_adapter()`) to prevent adapter bleed and preserve base model zero-shot answer generation style.
- **Dual-Branch Error Recovery:**
  - **Category B Recovery (Entity Switching):** If the verifier determines Candidate 2 matches the visual features better, routing switches to $e_{(2)}$.
  - **Category C Recovery (Multi-Section Context Expansion):** If Candidate 1 is visually confirmed, the entity is correct but the single selected section lacked evidence. The controller dynamically expands the prompt context by concatenating Candidate 1's Top-1 and Top-2 sections (`sec1 + "\n\n--- [Additional Section Evidence] ---\n\n" + sec2`).

---

## 📊 Benchmark Results

### 1. High-Fidelity Reproduction of OMGM (ACL 2025)
Evaluated on 1,000 representative InfoSeek test queries over the full, unpruned 100,000-article Wikipedia knowledge base:

| Pipeline Stage & Metric | OMGM Paper (Full InfoSeek) | Our Reproduction (1,000 Test Queries) | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Step 1: Coarse Entity Search** | | | | |
| Recall@1 | 52.6% | **49.50%** (495 / 1,000) | -3.10% | High Fidelity |
| Recall@5 | 73.9% | **71.50%** (715 / 1,000) | -2.40% | High Fidelity |
| Recall@10 | 80.0% | **78.80%** (788 / 1,000) | -1.20% | High Fidelity |
| Recall@20 (Search Space Ceiling) | **84.8%** | **83.40%** (834 / 1,000) | **-1.40%** | **Near-Exact Match** |
| **Step 2: Multimodal IT2IT Reranking** | | | | |
| Recall@1 | 64.0% | **59.10%** (591 / 1,000) | -4.90% | **+9.60% precision jump** |
| Recall@5 | 80.8% | **76.70%** (767 / 1,000) | -4.10% | High Fidelity |
| Recall@20 (Ceiling) | **84.8%** | **83.40%** (834 / 1,000) | **-1.40%** | Near-Exact Match |
| **Step 3: End-to-End VQA Generation** | | | | |
| VQA Accuracy (Official Metric) | **33.29%** | **34.90%** (349 / 1,000) | **+1.61%** | **Canonical Baseline Match** |

*Note on Evaluation Metric:* All VQA accuracy figures adhere to official OMGM evaluation (`check_match`), incorporating multi-alias target normalization and $\pm 10\%$ numerical tolerance.

---

### 2. Main End-to-End Evaluation Under Strict Zero Ground-Truth Leakage

| Model & Controller Configuration | VQA Accuracy | Gross Recoveries (Wrong $\to$ Right) | Degradations (Right $\to$ Wrong) | Net Change | Decision Win-Rate | Key Characteristics & Empirical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **OMGM Baseline** (Rigid Pipeline) | 34.90% (349) | 0 | 0 | -- | -- | Single-pass error cascade (40.8% candidate loss) |
| **Zero-Shot LLaVA-1.5-7B** | 34.90% (349) | 1 | 1 | +0 | 50.0% | Severe Option-(A) primacy bias (~96% collapse) |
| **Zero-Shot LLaVA-1.5-13B (Permissive)** | 34.20% (342) | 37 | 44 | -7 | 45.6% | Over-switching (360/508); uncalibrated coin-flip |
| **Zero-Shot LLaVA-1.5-13B (Conservative)**| 34.70% (347) | 17 | 19 | -2 | 47.2% | Precision is ~47% across both gating policies |
| **Ours: Fine-Tuned 7B + LoRA (Cat B)** | **36.00% (360)** | **27** | **16** | **+11** | **62.8%** | **Pareto Champion: 1.33% bias, +1.10% net gain, 44% faster** |
| **Ours: Fine-Tuned 13B + LoRA (Cat B)** | 35.60% (356) | 27 | 20 | +7 | 57.5% | Reverses 13B zero-shot deficit (-7 $\to$ +7, +14 turnaround) |
| **Ours: Fine-Tuned 7B + LoRA (Cat B+C)** | 35.10% (351) | 33 | 31 | +2 | 51.6% | Distractor dilution: 7B capacity bottleneck in long context |
| **Ours: Champion 13B + LoRA (Cat B+C)** | **36.30% (363)** | **44** | **30** | **+14** | **59.5%** | **Champion Model: New SOTA (+1.40%, 44 rescued queries)** |

---

### 3. The $2 \times 2$ Capacity & Verification Double Dissociation Matrix

Our comprehensive evaluation across model scales and task formulations reveals a fundamental **double dissociation**:

```
                       Task Formulation
                 Pairwise Entity         Multi-Section
                 Disambiguation (Cat B)  Expansion (Cat B+C)
               ┌───────────────────────┬───────────────────────┐
Backbone   7B  │  36.00% (+1.10% / +11)│  35.10% (+0.20% / +2) │
Scale          │     PARETO CHAMPION   │   Distractor Dilution │
               ├───────────────────────┼───────────────────────┤
          13B  │  35.60% (+0.70% / +7) │  36.30% (+1.40% / +14)│
               │   Slight Overconfidence│    CAPACITY CHAMPION  │
               └───────────────────────┴───────────────────────┘
```

1. **Why 7B Establishes the Pareto Frontier for Pairwise Verification (Cat B):**
   - Lightweight 7B fine-tuned with symmetric LoRA achieves remarkable visual agility and calibration (1.33% positional bias).
   - Incurs only **16 degradations** across 1,000 queries, delivering a **62.8% decision win-rate**.
   - Runs **44% faster** (3.24s vs. 4.67s/query) with **50% lower VRAM usage** than 13B.
2. **Why 13B is Structurally Essential for Multi-Section Context Expansion (Cat B+C):**
   - Concatenating Top-1 and Top-2 sections doubles prompt context to ~3,200 characters.
   - 7B suffers from "lost in the middle" distractor dilution (-6 net Cat C loss), frequently extracting structural headers like `"Family"` or `"Kingdom"` instead of specific answers.
   - 13B easily digests multi-section evidence (+8 net Cat C gain, 18 recoveries), driving total gross recoveries to **44 queries**.

---

### 4. Zero Ground-Truth Leakage & Generalization Proof

To verify that the verifier acquired generalizable visual adjudicative abilities rather than memorizing entity facts:
- **Image-Level Disjointness:** **100% strict disjointness** (0 out of 983 evaluation images were present in the training set).
- **Entity-Level Disjointness:** **88.0% (528 / 600 unique test entities)** were completely unseen during training.
- **Open-World Recovery Transfer:**
  - In 7B (Cat B), **85.19% (23 / 27)** of rescued queries involved previously unseen Wikipedia entities.
  - In Champion 13B (Cat B+C), **90.91% (40 / 44)** of rescued queries involved previously unseen Wikipedia entities.

---

## 🔬 Qualitative Case Studies (Rescued Queries)

Below are representative examples where OMGM failed due to rigid pipeline execution, but Adaptive Agentic Retrieval successfully recovered the ground-truth answer:

```
1. Query ID: infoseek_val_00000477
   Question:    "Where is the lake outflow to?"
   OMGM Pred:   "Lake Zürich" [Wrong - Candidate 1]
   LoRA Switch: (B) Lake Lucerne (Candidate 2 selected)
   Ours Pred:   "Reuss" (Ground Truth: Reuss (river)|Reuss|Reuss River) -> [RECOVERED]

2. Query ID: infoseek_val_00013636
   Question:    "Which body of water is this city located in or next to?"
   OMGM Pred:   "Caspian Sea" [Wrong - Candidate 1]
   LoRA Switch: (B) Kazan (Candidate 2 selected)
   Ours Pred:   "Volga" (Ground Truth: Volga River|Volga|Kazanka|River Volga) -> [RECOVERED]

3. Query ID: infoseek_val_00015575
   Question:    "What was the building or structure prior to the existence of this building?"
   OMGM Pred:   "No information provided." [Wrong - Candidate 1]
   LoRA Switch: (B) St. Peter's Basilica (Candidate 2 selected)
   Ours Pred:   "Old St. Peter's Basilica" (Ground Truth: Old St. Peter's Basilica) -> [RECOVERED]

4. Query ID: infoseek_val_00044711
   Question:    "Who designed this bridge?"
   OMGM Pred:   "Josef Václav Myslbek" [Wrong - Candidate 1]
   LoRA Switch: (B) Franz Joseph Bridge (Candidate 2 selected)
   Ours Pred:   "Rowland Mason Ordish" (Ground Truth: Rowland Mason Ordish) -> [RECOVERED]

5. Query ID: infoseek_val_00000289 (Category C Context Expansion)
   Question:    "how many metre is the wingspan of this bird?"
   OMGM Pred:   "132" [Hallucination - single introductory section lacked data]
   LoRA Action: Verified Candidate 1 as visually correct -> Expanded to Top-2 sections
   Ours Pred:   "1.32 meters" (Ground Truth: 1.34, within ±10% numerical tolerance) -> [RECOVERED]

6. Query ID: infoseek_val_00071549 (Category C Context Expansion)
   Question:    "What is this mountain named after?"
   OMGM Pred:   "Denali" [Wrong]
   LoRA Action: Verified Candidate 1 as visually correct -> Expanded to Top-2 sections
   Ours Pred:   "William McKinley" (Ground Truth: President McKinley|William McKinley) -> [RECOVERED]
```

---

## 📁 Repository Structure

```
agentic-vqa/
├── README.md                                           # Master project documentation
├── .gitignore                                          # Git ignore rules
│
├── my paper track/                                     # Paper draft, figures & research tracking
│   ├── experiment_and_paper_tracker.md                 # Complete experimental log, lineage & notes
│   ├── IEEE_Conference_Template/                       # LaTeX conference paper sources
│   │   ├── conference_101719.tex                       # Main paper LaTeX source
│   │   ├── refs.bib                                    # Complete bibliography
│   │   ├── agentic_vqa.pdf                             # Compiled camera-ready manuscript
│   │   └── figs/                                       # High-resolution vector & raster figures
│   │       ├── fig0_architecture_diagram.pdf           # System architecture overview
│   │       ├── fig1_failure_taxonomy_and_score_margin.pdf # Empirical failure audit & score margin
│   │       ├── fig2_verifier_diagnostic_and_debiasing.pdf # Zero-shot collapse vs. symmetric PEFT
│   │       └── fig3_scaling_matrix_double_dissociation.pdf # 2x2 double dissociation scaling matrix
│   └── paper works/
│       ├── generate_paper_figures.py                   # Matplotlib script generating publication figures
│       └── figs/                                       # Rendered publication figures
│
├── my paper implementation/                            # Core model codes, training & inference
│   ├── codes/
│   │   ├── build_verifier_dataset.py                   # Generates 6,000 symmetric PEFT training pairs
│   │   ├── colab_extract_train_images.py               # Selective extractor for required training images
│   │   ├── kaggle_agentic_controller.py (.ipynb)       # Zero-shot 7B controller
│   │   ├── kaggle_agentic_controller_llava13b.py (.ipynb) # Zero-shot 13B controller (permissive/conservative)
│   │   ├── fine tuning llava1.5 7b/
│   │   │   ├── kaggle_fine_tune_llava_verifier.py      # QLoRA fine-tuning script for LLaVA-1.5-7B
│   │   │   └── llava7b_finetuned_agentic_controller.py # End-to-end 7B agentic controller
│   │   └── fine tuning llava1.5 13b/
│   │       ├── kaggle_fine_tune_llava13b_verifier.py   # QLoRA fine-tuning script for LLaVA-1.5-13B
│   │       └── llava13b_finetuned_agentic_controller.py # End-to-end 13B controller (Cat B & Cat B+C)
│   └── results/                                        # Prediction outputs, logs & evaluation summaries
│       ├── controller fine tuned llava 7b/
│       │   ├── finetuned_final_predictions.json        # 1,000 query predictions (7B Cat B)
│       │   ├── finetuned_recovered_queries.json        # Detailed audit of 27 recovered queries (7B)
│       │   ├── finetuned_7b_cat_bc_predictions.json    # 1,000 query predictions (7B Cat B+C)
│       │   └── finetuned_llava 7b controlller.md       # Full execution log & metrics
│       └── controller fine tuned llava 13b/
│           ├── finetuned_13b_final_predictions.json    # 1,000 query predictions (13B Cat B)
│           ├── finetuned_13b_cat_bc_predictions.json   # 1,000 query predictions (Champion 13B Cat B+C)
│           ├── finetuned_13b_cat_bc_recovered_queries.json # 44 rescued queries audit
│           ├── finetuned_13b_cat_bc_results_summary.json # Machine-readable summary
│           └── finetuned_llava 13b controlller.md      # Full execution log & metrics
│
└── pre helper/                                         # OMGM baseline reproduction & Phase 0 failure audit
    ├── codes/
    │   ├── step 1/kaggle_step1_run_and_extract_images.py (.ipynb) # Step 1 coarse FAISS search
    │   ├── step 2/kaggle_step2_it2it_rerank.py (.ipynb) # Step 2 Q-Former IT2IT reranking
    │   ├── step 3/kaggle_step3_answer_generation.py (.ipynb) # Step 3 BGE selection + LLaVA VQA
    │   └── others/
    │       ├── kaggle_evidence_failure_analysis.py     # Audits Phase 0 failure taxonomy (Cat A-D)
    │       └── sanitize_wiki_images_for_kaggle.py      # Preprocesses Wikipedia candidate image links
    ├── files/
    │   ├── infoseek_test_sample_1000.csv               # 1,000 ground-truth test queries
    │   └── candidate_wiki_images_max3.csv              # 20,199 Wikipedia entity candidate image mappings
    └── res/
        ├── result_step2.json                           # Step 2 reranked candidate pools & scores
        ├── step3_predictions.json                      # OMGM baseline predictions (34.90% accuracy)
        ├── step3_reranked_sections.json                # Top selected sections from BGE reranker
        └── phase0_failure_analysis.json                # Complete Category A-D failure classification
```

---

## ⚙️ Hardware Requirements & Engineering Optimizations

The framework is optimized to run seamlessly in resource-constrained environments (e.g., standard free-tier Kaggle/Colab GPUs with 16GB VRAM):

| Task | Minimum GPU | Recommended GPU | Quantization | Peak VRAM | Runtime (1,000 Queries) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Step 1 (Coarse FAISS)** | CPU / 1x T4 | 1x NVIDIA T4 (16GB) | FP16 | ~4.0 GB | ~8 min |
| **Step 2 (Q-Former Rerank)**| 1x NVIDIA T4 (16GB) | 1x NVIDIA T4 (16GB) | FP16 (Batch 32) | ~7.5 GB | ~35 min |
| **Step 3 (OMGM Baseline)** | 1x NVIDIA T4 (16GB) | 1x NVIDIA T4 (16GB) | 4-bit NF4 | ~8.0 GB | ~28 min |
| **LoRA Fine-Tuning (7B)** | 1x NVIDIA T4 (16GB) | 2x NVIDIA T4 (16GB) | 4-bit NF4 QLoRA | ~11.5 GB / GPU | ~4.2 hours (1 epoch) |
| **LoRA Fine-Tuning (13B)**| 2x NVIDIA T4 (16GB) | 2x NVIDIA T4 (16GB) | 4-bit NF4 QLoRA | ~13.8 GB / GPU | ~7.2 hours (1 epoch) |
| **Agentic Controller (7B)**| 1x NVIDIA T4 (16GB) | 1x NVIDIA T4 / 2x T4 | 4-bit NF4 | ~4.1 GB / GPU | ~27 min (1.65s / query) |
| **Champion Controller (13B)**| 2x NVIDIA T4 (16GB)| 2x NVIDIA T4 / RTX 3090 | 4-bit NF4 | ~7.8 GB / GPU | ~52 min (3.14s / query) |

### Engineering Breakthroughs:
1. **Python 3.13 & Modern Transformers Bridge:** Resolved legacy LAVIS / BLIP-2 incompatibilities with modern Hugging Face versions (`transformers >= 4.41.0`) under Python 3.13 through dynamic forward chunking bridges, pruning stub restoration, tied-weights descriptors, and vocabulary expansion synchronization (`[DEC]` token resize).
2. **Two-Stage VRAM Sequencing:** In Step 3, `BAAI/bge-reranker-v2-m3` is fully executed and explicitly evacuated from GPU memory before instantiating the 4-bit VLM generator, preventing Out-Of-Memory (OOM) faults.
3. **Dual-Mode Adapter Isolation:** Verifier LoRA adapters are dynamically attached for single-pass entity adjudication and cleanly detached (`disable_adapter()`) prior to answer synthesis, ensuring zero adapter bleed.

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/agentic-vqa.git
cd agentic-vqa

# Create and activate a clean conda environment
conda create -n agentic-vqa python=3.10 -y
conda activate agentic-vqa

# Install PyTorch with CUDA support (adjust CUDA version if necessary)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers accelerate bitsandbytes peft faiss-gpu pandas numpy Pillow tqdm
```

---

### 2. Dataset Preparation

Download the required benchmark files:
- **100k Wikipedia Knowledge Base:** Download `wiki_100_dict_v4.json` from the [OMGM Hugging Face repository](https://huggingface.co/datasets/BByrneLab/Infoseek-train-with-EvaClip-retrieval-top5).
- **1,000 InfoSeek Test Queries:** Included in [`pre helper/files/infoseek_test_sample_1000.csv`](pre%20helper/files/infoseek_test_sample_1000.csv).
- **Wikipedia Entity Images:** Extracted and mapped in [`pre helper/files/candidate_wiki_images_max3.csv`](pre%20helper/files/candidate_wiki_images_max3.csv).

---

### 3. Step-by-Step Pipeline Execution

#### Step 3.1: Run OMGM Baseline Reproduction
```bash
# 1. Execute Step 1 Coarse Retrieval (produces top-20 candidates via CLIP + FAISS)
python "pre helper/codes/step 1/kaggle_step1_run_and_extract_images.py"

# 2. Execute Step 2 Multimodal Reranking (Q-Former IT2IT cross-attention)
python "pre helper/codes/step 2/kaggle_step2_it2it_rerank.py"

# 3. Execute Step 3 Section Reranking & Answer Generation (LLaVA-1.5-7B zero-shot)
python "pre helper/codes/step 3/kaggle_step3_answer_generation.py"

# 4. Run Phase 0 Evidence-First Failure Audit (Cat A-D classification)
python "pre helper/codes/others/kaggle_evidence_failure_analysis.py"
```

#### Step 3.2: Build Symmetric PEFT Calibration Dataset
```bash
# Generates 6,000 balanced pairs (50% Target A, 50% Target B) from OVEN training data
python "my paper implementation/codes/build_verifier_dataset.py"
```

#### Step 3.3: Fine-Tune the Visual Verifier (QLoRA)
```bash
# Fine-tune LLaVA-1.5-7B Verifier Adapter (1 epoch, ~4.25 hours on 2x T4)
python "my paper implementation/codes/fine tuning llava1.5 7b/kaggle_fine_tune_llava_verifier.py"

# Or fine-tune LLaVA-1.5-13B Verifier Adapter (1 epoch, ~7.18 hours on 2x T4)
python "my paper implementation/codes/fine tuning llava1.5 13b/kaggle_fine_tune_llava13b_verifier.py"
```

#### Step 3.4: Run the Adaptive Agentic Retrieval Controller

##### Option A: Pareto Champion (Fine-Tuned 7B, Category B Recovery)
```bash
python "my paper implementation/codes/fine tuning llava1.5 7b/llava7b_finetuned_agentic_controller.py"
```
*Evaluates the 508 ambiguity-zone queries ($\Delta < 0.025$). Achieves **36.00% accuracy (+1.10% / +11 net queries, 62.8% win-rate)** in ~27 minutes.*

##### Option B: Champion Model (Fine-Tuned 13B, Category B + C Recovery)
```bash
python "my paper implementation/codes/fine tuning llava1.5 13b/llava13b_finetuned_agentic_controller.py"
```
*Executes dual-branch recovery: entity switching for Category B and Top-2 multi-section expansion for Category C. Achieves **36.30% accuracy (+1.40% / +14 net queries, 44 gross recoveries)**.*

---

### 4. Reproducing Publication Figures

All vector figures featured in the paper can be reproduced using:
```bash
python "my paper track/paper works/generate_paper_figures.py"
```
Rendered PDF and PNG outputs will be generated in `my paper track/paper works/figs/` and synced with `my paper track/IEEE_Conference_Template/figs/`.

---

## 📝 Citation

If you find this work, repository, or experimental methodology useful in your research, please cite our paper:

```bibtex
@inproceedings{abir2026adaptive,
  title={Adaptive Agentic Retrieval: Breaking the Rigid Pipeline Bottleneck in Knowledge-Based Visual Question Answering},
  author={Abir, Shahad and Asif, Abdullah Mohammed},
  booktitle={IEEE Conference Proceedings},
  year={2026},
  pages={1--6},
  url={my paper track/IEEE_Conference_Template/conference_101719.pdf}
}
```

---

## 📄 License & Acknowledgments

- This project is released under the [MIT License](LICENSE).
- We gratefully acknowledge the authors of **OMGM** (Lin et al., ACL 2025), **InfoSeek** (Yang et al., EMNLP 2023), **OVEN** (Hu et al., CVPR 2023), and **LLaVA** (Liu et al., CVPR 2024) for releasing their open datasets, models, and codebases.
