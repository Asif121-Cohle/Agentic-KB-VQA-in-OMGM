CELL 1: ENVIRONMENT SETUP, SEEDS & PATH CONFIGURATION (13B)
======================================================================
Deterministic seed set to 42.
CUDA available: True
Detected 2 CUDA Device(s):
  GPU [0]: Tesla T4 | VRAM: 14.56 GB
  GPU [1]: Tesla T4 | VRAM: 14.56 GB

Resolved Dataset & Model Paths:
  Step 2 Result  : /kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/result_step_2.json  (exists: True)
  Step 3 Baseline: /kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/step3_predictions.json  (exists: True)
  Sample CSV     : /kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/infoseek_test_sample_1000.csv  (exists: True)
  KB JSON        : /kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/infoseek_100k_wiki/wiki_100_dict_v4.json  (exists: True)
  Query Images   : /kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/infoseek_val_1000_images/infoseek_val_1000_images  (exists: True)
  LoRA 13B Adapter: /kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/llava_13b_verifier_lora  (exists: True)
  Tau Threshold  : 0.025
  Output Dir     : /kaggle/working/agentic_results_finetuned_13b


Evaluation metric loaded: Canonical OMGM check_match with +/- 10% numeric tolerance.

Fetching 6 files: 100%
 6/6 [01:51<00:00, 10.60s/it]
Loading weights: 100%
 758/758 [01:39<00:00, 163.94it/s]
generation_config.json: 100%
 141/141 [00:00<00:00, 12.7kB/s]
  After base model — GPU [0]: 3.98 GB allocated
  After base model — GPU [1]: 3.98 GB allocated

3. Attaching Fine-Tuned 13B LoRA Adapter from: '/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/llava_13b_verifier_lora'
   LoRA 13B adapter attached successfully.
  After LoRA attach — GPU [0]: 4.08 GB allocated
  After LoRA attach — GPU [1]: 4.12 GB allocated

4. Verification: Confirm LoRA 13B Adapter is Active (Inference Mode)
   Active adapter(s): ['default']
   Adapter type     : PeftType.LORA
   LoRA rank (r)    : 16
   LoRA target mods : {'o_proj', 'up_proj', 'q_proj', 'v_proj', 'down_proj', 'gate_proj', 'k_proj'}
   Adapter weights  : /kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/llava_13b_verifier_lora/adapter_model.safetensors (exists: True)
   NOTE: requires_grad=False is CORRECT for inference — LoRA hooks are active.
   NOTE: generate_answer() will use disable_adapter() to restore pure base model.

Inference helpers loaded:
  - generate_answer()            -> base_model (no LoRA, OMGM-style VQA)
  - ask_visual_choice_finetuned() -> verifier_model (13B base + LoRA, single-pass)

CELL 7: DATA LOADING & BASELINE ANALYSIS
======================================================================
Loaded: 1,000 Step 2 entries | 1,000 Step 3 entries | 1,000 sample metadata rows

Baseline OMGM Accuracy: 34.90% (349/1000)
Score Margin — Mean: 0.0362 | Min: 0.0000 | Max: 0.1664
Ambiguous Queries (Δ < 0.025): 508 / 1000 (50.8%)
Fast-Path Queries (Δ ≥ 0.025): 492 / 1000 (49.2%)

CELL 9: PILOT RUN — FINE-TUNED 13B VERIFIER ON 3 QUERIES
======================================================================
-----------------------------------------------------------------
Query ID   : infoseek_val_00000075
Question   : What is the closest parent taxonomy of this bird?
Δ (S1-S2)  : 0.0220  [Ambiguous: Δ < 0.025]
Candidate 1: Pomarine jaeger
Candidate 2: Pink-footed shearwater
LoRA Choice: -> A (FineTuned13B→Cand1 | gen: (A) Pomarine jaeger)
Baseline Pred: 'Great skua' | GT: 'Fulmar|Fulmarus'
Baseline Correct: False
-----------------------------------------------------------------
Query ID   : infoseek_val_00000228
Question   : What is the conservation status of this animal? (The status is assigned by the international union for conservation of nature. Choose one among Endangered,Least Concern,Critically Endangered,extinct species,extinct in the wild,Vulnerable,Near Threatened,Data Deficient)
Δ (S1-S2)  : 0.0011  [Ambiguous: Δ < 0.025]
Candidate 1: Jaguar
Candidate 2: Leopard
LoRA Choice: -> B (FineTuned13B→Cand2 | gen: (B) Leopard)
Baseline Pred: 'Endangered' | GT: 'Near Threatened|LR/nt|NT'
Baseline Correct: False
-----------------------------------------------------------------
Query ID   : infoseek_val_00000289
Question   : how many metre is the wingspan of this bird?
Δ (S1-S2)  : 0.0042  [Ambiguous: Δ < 0.025]
Candidate 1: Glaucous gull
Candidate 2: European herring gull
LoRA Choice: -> A (FineTuned13B→Cand1 | gen: (A) Glaucous gull)
Baseline Pred: '132' | GT: '1.34'
Baseline Correct: False
-----------------------------------------------------------------

======================================================================
CELL 10: AGENTIC CONTROLLER LOOP (FINE-TUNED 13B LoRA VERIFIER)
======================================================================
Processing 508 / 508 remaining ambiguous queries...
Fine-Tuned 13B Agentic Loop: 100%
 508/508 [37:15<00:00,  4.67s/it]
  Checkpoint @ 50/508 | Elapsed: 3.2 min | Switches: 12
  Checkpoint @ 100/508 | Elapsed: 6.7 min | Switches: 28
  Checkpoint @ 150/508 | Elapsed: 10.7 min | Switches: 53
  Checkpoint @ 200/508 | Elapsed: 14.3 min | Switches: 72
  Checkpoint @ 250/508 | Elapsed: 18.0 min | Switches: 92
  Checkpoint @ 300/508 | Elapsed: 21.7 min | Switches: 114
  Checkpoint @ 350/508 | Elapsed: 25.2 min | Switches: 131
  Checkpoint @ 400/508 | Elapsed: 29.3 min | Switches: 156
  Checkpoint @ 450/508 | Elapsed: 32.8 min | Switches: 172
  Checkpoint @ 500/508 | Elapsed: 36.6 min | Switches: 192
  Checkpoint @ 508/508 | Elapsed: 37.3 min | Switches: 197

Agentic Loop Finished in 2235.6s.
Visual switches to Candidate 2: 197
Category C section refreshes:   0

======================================================================
CELL 11: FINAL ZERO-LEAKAGE EVALUATION (FINE-TUNED 13B CONTROLLER)
======================================================================

Metric                              | Baseline (OMGM)      | Ours (LoRA 13B)      | Delta
------------------------------------------------------------------------------------------
Overall VQA Accuracy                |  34.90% (349/1000) |  35.60% (356/1000)   | +0.70% (+7)
Ambiguous Queries Triggered         | N/A                  | 508                  | 50.8%
Gross Recoveries (Wrong->Right)     | 0 (rigid)            | 27                   | +27
Degradations (Right->Wrong)         | 0 (rigid)            | 20                   | -20
Net Improvement                     | 0                    | +7 queries            | Honest Net
Decision Win-Rate on Flips          | --                   |  57.45%              | (27/47)
McNemar Chi-Squared (p-val)         | --                   |  0.766               | Chi2 stat
Avg. Generator Calls/Query          | 1.00                 | 1.51                   | +0.51 overhead
==========================================================================================


======================================================================
CELL 13: CATEGORY B + C CONTROLLER (MULTI-SECTION CONTEXT EXPANSION)
======================================================================
Executing Category B+C Controller on 508 ambiguous queries...
Category B+C Controller: 100%
 508/508 [52:18<00:00,  6.21s/it]

Category B+C Controller Finished in 52.3 minutes.
  Switches to Candidate 2 (Cat B) : 195
  Multi-Section Expansions (Cat C): 313
Saved: /kaggle/working/agentic_results_finetuned_13b/finetuned_13b_cat_bc_predictions.json


======================================================================
CELL 14: CATEGORY B + C SCORECARD
======================================================================

Metric                              | Baseline (OMGM)      | Ours (Cat B+C 13B)   | Delta          
------------------------------------------------------------------------------------------------
Overall VQA Accuracy                | 34.90% (349/1000)     | 36.30% (363/1000)     | ++1.40% (+14)
Gross Recoveries (Wrong->Right)     | 0 (rigid)            | 44                   | +44
Degradations (Right->Wrong)         | 0 (rigid)            | 30                   | -30
Net Improvement                     | 0                    | +14 queries           | Honest Net
Decision Win-Rate on Flips          | --                   | 59.46%               | (44/74)
================================================================================================
