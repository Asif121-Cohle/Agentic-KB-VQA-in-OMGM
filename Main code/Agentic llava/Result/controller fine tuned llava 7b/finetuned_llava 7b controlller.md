  After base model — GPU [0]: 4.09 GB allocated
  After base model — GPU [1]: 4.68 GB allocated

3. Attaching Fine-Tuned LoRA Adapter from: '/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/llava_verifier_lora'
   LoRA adapter attached successfully.
  After LoRA attach — GPU [0]: 4.15 GB allocated
  After LoRA attach — GPU [1]: 4.78 GB allocated

4. Verification: Confirm LoRA Adapter is Active (Inference Mode)
   Total model params (base + LoRA deltas): 3,706,279,936
   Estimated LoRA delta params: 0
   Active adapter(s): ['default']
   Adapter type: PeftType.LORA
   Adapter weights file: /kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/llava_verifier_lora/adapter_model.safetensors (exists: True)
   NOTE: requires_grad=False is CORRECT for inference — LoRA weights are active and applied.
Inference helpers loaded:
  - generate_answer()            → base_model (no LoRA, OMGM-style VQA)
  - ask_visual_choice_finetuned() → verifier_model (base + LoRA, single-pass)

======================================================================
CELL 7: DATA LOADING & BASELINE ANALYSIS
======================================================================
Loaded: 1,000 Step 2 entries | 1,000 Step 3 entries | 1,000 sample metadata rows

Baseline OMGM Accuracy: 34.90% (349/1000)
Score Margin — Mean: 0.0362 | Min: 0.0000 | Max: 0.1664
Ambiguous Queries (Δ < 0.025): 508 / 1000 (50.8%)
Fast-Path Queries (Δ ≥ 0.025): 492 / 1000 (49.2%)

======================================================================
CELL 9: PILOT RUN — FINE-TUNED VERIFIER ON 3 QUERIES
======================================================================
-----------------------------------------------------------------
Query ID   : infoseek_val_00000075
Question   : What is the closest parent taxonomy of this bird?
Δ (S1-S2)  : 0.0220  [Ambiguous: Δ < 0.025]
Candidate 1: Pomarine jaeger
Candidate 2: Pink-footed shearwater
LoRA Choice: → A (FineTuned→Cand1 | gen: (A) Pomarine jaeger)
Baseline Pred: 'Great skua' | GT: 'Fulmar|Fulmarus'
Baseline Correct: False
-----------------------------------------------------------------
Query ID   : infoseek_val_00000228
Question   : What is the conservation status of this animal? (The status is assigned by the international union for conservation of nature. Choose one among Endangered,Least Concern,Critically Endangered,extinct species,extinct in the wild,Vulnerable,Near Threatened,Data Deficient)
Δ (S1-S2)  : 0.0011  [Ambiguous: Δ < 0.025]
Candidate 1: Jaguar
Candidate 2: Leopard
LoRA Choice: → B (FineTuned→Cand2 | gen: (B) Leopard)
Baseline Pred: 'Endangered' | GT: 'Near Threatened|LR/nt|NT'
Baseline Correct: False
-----------------------------------------------------------------
Query ID   : infoseek_val_00000289
Question   : how many metre is the wingspan of this bird?
Δ (S1-S2)  : 0.0042  [Ambiguous: Δ < 0.025]
Candidate 1: Glaucous gull
Candidate 2: European herring gull
LoRA Choice: → B (FineTuned→Cand2 | gen: (B) European herring gull)
Baseline Pred: '132' | GT: '1.34'
Baseline Correct: False
-----------------------------------------------------------------

======================================================================
CELL 10: AGENTIC CONTROLLER LOOP (FINE-TUNED LORA VERIFIER)
======================================================================
Processing 508 ambiguous queries using fine-tuned verifier...
Fine-Tuned Agentic Loop:  10%|▉         | 50/508 [02:37<21:54,  2.87s/it]
  Checkpoint @ 50/508 | Elapsed: 2.6 min | Switches: 17
Fine-Tuned Agentic Loop:  20%|█▉        | 100/508 [05:10<20:50,  3.07s/it]
  Checkpoint @ 100/508 | Elapsed: 5.2 min | Switches: 36
Fine-Tuned Agentic Loop:  30%|██▉       | 150/508 [07:55<17:36,  2.95s/it]
  Checkpoint @ 150/508 | Elapsed: 7.9 min | Switches: 58
Fine-Tuned Agentic Loop:  39%|███▉      | 200/508 [10:40<15:18,  2.98s/it]
  Checkpoint @ 200/508 | Elapsed: 10.7 min | Switches: 80
Fine-Tuned Agentic Loop:  49%|████▉     | 250/508 [13:29<16:15,  3.78s/it]
  Checkpoint @ 250/508 | Elapsed: 13.5 min | Switches: 106
Fine-Tuned Agentic Loop:  59%|█████▉    | 300/508 [16:20<14:37,  4.22s/it]
  Checkpoint @ 300/508 | Elapsed: 16.3 min | Switches: 133
Fine-Tuned Agentic Loop:  69%|██████▉   | 350/508 [19:00<08:37,  3.28s/it]
  Checkpoint @ 350/508 | Elapsed: 19.0 min | Switches: 154
Fine-Tuned Agentic Loop:  79%|███████▊  | 400/508 [21:48<06:30,  3.61s/it]
  Checkpoint @ 400/508 | Elapsed: 21.8 min | Switches: 178
Fine-Tuned Agentic Loop:  89%|████████▊ | 450/508 [24:20<02:33,  2.64s/it]
  Checkpoint @ 450/508 | Elapsed: 24.3 min | Switches: 195
Fine-Tuned Agentic Loop:  98%|█████████▊| 500/508 [26:54<00:26,  3.31s/it]
  Checkpoint @ 500/508 | Elapsed: 26.9 min | Switches: 213
Fine-Tuned Agentic Loop: 100%|██████████| 508/508 [27:25<00:00,  3.24s/it]

Agentic Loop Finished in 1645.2s (3.24s per query).
Visual switches to Candidate 2: 219
Category C section refreshes:   0

======================================================================
CELL 11: FINAL ZERO-LEAKAGE EVALUATION (FINE-TUNED CONTROLLER)
======================================================================

Metric                              | Baseline (OMGM)      | Ours (LoRA Verifier) | Delta
------------------------------------------------------------------------------------------
Overall VQA Accuracy                |  34.90% (349/1000) |  36.00% (360/1000)   | +1.10% (+11)
Ambiguous Queries Triggered         | N/A                  | 508                  | 50.8%
Gross Recoveries (Wrong→Right)      | 0 (rigid)            | 27                   | +27
Degradations (Right→Wrong)          | 0 (rigid)            | 16                   | -16
Net Improvement                     | 0                    | +11 queries            | Honest Net
Avg. Generator Calls/Query          | 1.00                 | 1.51                   | +0.51 overhead
==========================================================================================
{
  "model": "LLaVA-1.5-7B + LoRA Verifier (Fine-Tuned, Symmetric 6k Pairs)",
  "lora_adapter": "/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/llava_verifier_lora",
  "tau": 0.025,
  "total_queries": 1000,
  "baseline_accuracy": 34.9,
  "agentic_accuracy": 36.0,
  "accuracy_improvement": 1.1,
  "net_recovered_queries": 11,
  "gross_recovered_queries": 27,
  "degraded_queries": 16,
  "preserved_correct_queries": 333,
  "ambiguous_queries_triggered": 508,
  "trigger_rate_pct": 50.8,
  "switched_to_b_count": 219,
  "cat_c_refreshed_count": 0,
  "average_calls_per_query": 1.51,
  "loop_time_minutes": 27.42
}