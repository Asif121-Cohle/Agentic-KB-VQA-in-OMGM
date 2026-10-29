# A²MR: Adaptive Agentic Multimodal Retrieval for KB-VQA

> **Adaptive Agentic Multimodal Retrieval Breaking the Rigid Pipeline Bottleneck in Knowledge-Based Visual Question Answering**

---

## 📌 Overview

**Knowledge-Based Visual Question Answering (KB-VQA)** requires an AI system to inspect an image and consult external knowledge (such as Wikipedia) to answer questions that cannot be answered from visual pixels alone.

State-of-the-art systems like **OMGM (ACL 2025)** use a rigid, multi-stage retrieval pipeline:
1. **Stage 1 (Coarse Retrieval):** CLIP + FAISS search over 100k Wikipedia summaries $\to$ Top-20 candidate entities.
2. **Stage 2 (Multimodal Reranking):** BLIP-2 Q-Former cross-attention $\to$ Top-1 entity.
3. **Stage 3 (Fine-Grained Selection & Generation):** BGE passage reranking + LLaVA answer generation.

### The Problem
This sequential cascade is **rigid**. If the correct entity is ranked #2 by an extremely tiny margin at Stage 2, it is permanently discarded with **no mechanism for uncertainty detection, verification, or recovery**. Our empirical audit of 1,000 InfoSeek test queries revealed that **40.8% of all queries fail due to premature candidate elimination**, even though the needed knowledge was already present in the retrieved candidate pool.

### The Solution: A²MR
**A²MR** introduces a lightweight, rule-based **Adaptive Agentic Controller** that detects uncertainty on the fly:
- **Fast-Path (Confident Queries):** Follows the standard pipeline with **0 extra compute overhead**.
- **Ambiguity-Path (High-Risk Queries):** When the top-2 score gap collapses ($\Delta < \tau = 0.025$), the controller triggers dynamic pairwise visual verification and evidence expansion to rescue mistakenly discarded candidates.

```
                  Query Image + Question
                            │
               [Stage 1: Coarse Retrieval] (CLIP + FAISS)
                            │
              [Stage 2: Multimodal Reranking] (Q-Former)
                            │
               Is score margin Δ = S1 - S2 ≥ 0.025?
                       /                 \
             YES (Confident)          NO (Ambiguous)
                    /                         \
        [Fast-Path Execution]          [Agentic Controller]
     Standard BGE Section + VLM      Symmetric PEFT Verifier
                    \                         /
                     └───► Final Answer ◄────┘
```

---

## 🚀 Key Results & Discoveries

Evaluated on the **InfoSeek benchmark (1,000 test queries against the full 100,000 Wikipedia articles)** under strict **zero ground-truth leakage**:

| Pipeline / Model Configuration | VQA Accuracy | Net Recovered Queries | Key Mechanism |
| :--- | :---: | :---: | :--- |
| **Baseline OMGM** (Fixed Pipeline) | 34.90% | — | Rigid single-pass cascade |
| **A²MR: LLaVA-1.5-7B + LoRA Verifier** | **36.00%** | **+11 queries** (+1.10%) | Fast, Pareto-optimal pairwise verification |
| **A²MR: LLaVA-1.5-13B + LoRA Verifier (Champion)** | **36.30%** | **+14 queries** (+1.40%) | Verification + multi-section evidence expansion (44 rescued queries) |

### Core Scientific Findings
1. **Score-Margin Uncertainty Signal:** When Stage 2 predictions are correct, the average score margin $\Delta = S_1 - S_2$ is $0.0448$. When an entity error occurs, this margin collapses by $2.9\times$ to $0.0157$, providing an intrinsic, zero-cost threshold ($\tau = 0.025$).
2. **The Zero-Shot VLM Bottleneck:** Off-the-shelf zero-shot VLMs fail as verifiers due to severe positional primacy bias (favoring Option A ~21.7% of the time, resulting in coin-flip accuracy).
3. **Symmetric PEFT Calibration:** Training a lightweight LoRA adapter ($r=16$) on symmetric pairwise entity examples virtually eliminates positional bias ($21.67\% \to 1.33\%$) and unlocks genuine, net-positive error recovery.

---

## 📂 Repository Structure

```
Agentic KB-VQA/
├── Stage 1-Coarse/             # CLIP image embedding & FAISS search over Wikipedia summaries
│   └── agentic-vqa-step-1-for-wiki-imgs.ipynb
│
├── Stage 2-Hybrid/             # Q-Former multimodal reranking (Image + Question cross-attention)
│   ├── agentic-vqa-step-2-reranking.ipynb
│   └── result.json             # Reranked top candidates & confidence scores
│
├── Stage 3-FineGrained/        # Baseline section reranking (BGE) and answer generation
│   ├── agentic-vqa-step-3-topsection-generator.ipynb
│   ├── phase0_failure_analysis.json   # 1,000-query empirical error taxonomy
│   └── step3_predictions.json
│
├── Main code/                  # Core Agentic Controller & Adapters
│   ├── Agentic llava/
│   │   ├── Code/
│   │   │   ├── kaggle_agentic_controller.py / .ipynb      # Main controller implementation
│   │   │   ├── build_verifier_dataset.py                  # Symmetric pair generation
│   │   │   ├── fine tuning llava1.5 7b/                   # LoRA fine-tuning for 7B verifier
│   │   │   └── fine tuning llava1.5 13b/                  # LoRA fine-tuning for 13B verifier
│   │   └── Result/                                        # Prediction logs & evaluation metrics
│   └── Agentic Qween/          # Qwen2.5-VL agentic adaptation and experiments
│       └── agentic-omgm.ipynb
│
├── Latex Resources/            # Full academic paper source & tracking
│   ├── IEEE_Conference_Template/conference_101719.tex     # Full LaTeX manuscript
│   ├── IEEE_Conference_Template/figs/                     # Architecture & result figures
│   └── experiment_and_paper_tracker.md                    # Detailed research tracker
│
└── Supplementary/              # Background documents, setup guide, and OMGM specifications
```

---

## 🛠️ Quickstart

### Prerequisites & Environment
Designed to run on standard single-GPU environments (e.g., Kaggle free-tier NVIDIA T4 / P100 16GB VRAM):
- Python 3.10+ (includes compatibility patches for newer environments)
- PyTorch with CUDA support
- Hugging Face `transformers`, `peft`, `bitsandbytes`, `accelerate`

### Running the Agentic Controller
1. **Inspect / Run the Controller Notebook:**
   Open `Main code/Agentic llava/Code/kaggle_agentic_controller.ipynb` or execute the script:
   ```bash
   python "Main code/Agentic llava/Code/kaggle_agentic_controller.py"
   ```
2. **Fine-Tuning the Visual Verifier (LoRA PEFT):**
   ```bash
   python "Main code/Agentic llava/Code/fine tuning llava1.5 7b/kaggle_fine_tune_llava_verifier.py"
   ```
3. **Evaluating Results:**
   Check the summary files generated in `Main code/Agentic llava/Result/` for accuracy metrics, win rates, and recovery breakdowns.

---

## 📄 Citation & Reference

If you find this work or codebase helpful in your research, please refer to the manuscript located in `Latex Resources/IEEE_Conference_Template/`:

> **A²MR: Adaptive Agentic Retrieval Breaking the Rigid Pipeline Bottleneck in Knowledge-Based Visual Question Answering**  
> *Department of Computer Science and Engineering, Chittagong University of Engineering and Technology (CUET)*.
