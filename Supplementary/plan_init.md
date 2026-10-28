# Project Plan: Agentic Retrieval Controller for Knowledge-Based VQA

A reference document to hand to anyone (collaborator, future-you, or an AI assistant) so they understand the full scope and can start executing without re-deriving context from scratch.

---

## 1. What We're Building On

**Base paper:** *OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval* (ACL 2025 Main, arXiv:2505.07879)

**What OMGM does:** A fixed 3-step retrieval pipeline for Knowledge-Based Visual Question Answering (KB-VQA) — given a query image + question, it retrieves relevant Wikipedia knowledge to answer questions that require external knowledge beyond what's visible in the image.

```
Step 1 (Coarse): query image → CLIP → FAISS search over
                  entity summaries → top-20 candidate entities

Step 2 (Hybrid):  top-20 candidates → Q-Former multimodal
                  reranker (image+question vs image+sections)
                  → top-1 entity

Step 3 (Fine):    top-1 entity's sections → text reranker
                  → best section → LLM/MLLM generates answer
```

**Key limitation we're targeting:** This pipeline is **rigid**. Every query goes through all 3 steps identically, in the same order, exactly once. There is no mechanism to detect uncertainty, verify an ambiguous choice, retry with a different strategy, or stop early. If Step 2 picks the wrong entity, that error propagates forward with no recovery — the paper itself does not analyze this failure mode.

**OMGM's code/weights status:** Official repo at `github.com/ChaoLinAViy/OMGM` released full code + pretrained checkpoints (FAISS indexes, trained Q-Former reranker, fine-tuned LLaVA generator). We are **not retraining any of their components** — we use their released weights as frozen tools.

---

## 2. Our Core Research Question

> *Does OMGM's fixed pipeline fail in ways that an adaptive, agentic control flow could recover from — and if so, how much, and at what efficiency cost?*

This is intentionally framed as an empirical question first, method second. The plan below starts with **measuring the failure modes before designing the fix**, so the method is evidence-driven rather than "this sounds like a good idea."

---

## 3. Proposed Method (Summary)

We keep OMGM's trained components exactly as released, but **treat them as callable tools** rather than fixed pipeline stages, and wrap them in a lightweight, **rule-based** (not learned, not LLM-brained) agent controller that can loop back, branch, and stop adaptively.

```
Frozen tools (OMGM's released weights, inference-only):
├── tool_coarse_search_clip(image)
│       → Step 1 as OMGM does it
├── tool_coarse_search_fused(image, question)      [NEW USE]
│       → reuses OMGM's Q-Former — but for SEARCH,
│         not reranking. OMGM never does this.
├── tool_rerank(image, question, candidates)
│       → Step 2 as OMGM does it
├── tool_verify(image, entity)                      [NEW]
│       → MLLM prompt comparing query image vs
│         candidate entity's image — OMGM has no
│         verification step at all
├── tool_section_select(entity, question)
│       → Step 3 section picking, as OMGM does it
└── tool_generate(image, section, question)
        → Step 3 generation, as OMGM does it

Agent controller (our contribution — simple, rule-based):
├── Confidence signal = score gap between top-1 and top-2
│   candidates at each stage (cheap to compute, no training)
├── If gap below threshold → trigger extra action:
│   ├── ambiguous Step 1 → re-search using fused tool
│   ├── ambiguous Step 2 → run tool_verify
│   └── verify fails → retry search, or check top-2 entities
│       fully through Step 3 and compare
└── Otherwise → proceed straight through, same cost as OMGM
```

**Why rule-based, not a trained/learned policy or LLM-as-brain:**
- A trained policy needs labeled "best action" data we don't have and would be expensive to generate (ruled out for time/compute budget).
- An LLM making every routing decision means 3-5+ extra LLM calls per query — too slow/expensive at our scale on Kaggle T4.
- Thresholds are cheap, fully reproducible, and — critically — isolate our claim: *the novelty is in the control-flow structure itself (loop-back, verification, branching), not in how well-tuned the policy is.*

**Why this is not "just a wrapper":** every new action (fused search at Step 1, verification, top-2 comparison, retry) is something OMGM's architecture **structurally cannot do at all**, not an optimization of something it already does.

---

## 4. The Evidence-First Step: Failure Analysis (Do This Before Building Anything)

This is Phase 0 and it is not optional — it's what turns this from "an idea" into "a finding."

```
Procedure:
1. Run OMGM's released pipeline (frozen weights) end-to-end
   on our InfoSeek query subset (see §6)
2. Collect every question OMGM answers incorrectly
3. For each failure, classify into:

   Category A — Wrong entity never in Step 1's top-20
                (retrieval ceiling failure)
   Category B — Correct entity WAS in top-20, but Step 2
                reranker picked the wrong one
                (reranking failure — closest visual match
                 beats correct match)
   Category C — Correct entity selected, wrong section
                chosen in Step 3
   Category D — Correct entity + correct section, but
                generator still answered wrong
                (pure reasoning/generation failure)

4. Report the breakdown as a table/chart — this becomes
   the paper's motivation section
```

**Why this matters for the method:** the breakdown tells us which agent actions are actually worth building.
- If **Category B dominates** → verification + top-2 comparison is well-motivated (our current plan assumes this, but we confirm it empirically first).
- If **Category A dominates** → the fused-search retry action matters more than verification.
- If **Category D dominates** → agentic retrieval won't help much at all, and that itself is a valid (if less exciting) finding to report honestly.

---

## 5. Dataset Decision

**Using InfoSeek only** (not E-VQA, not both). Rationale already settled:
- InfoSeek's released FAISS index (287MB) and 100K knowledge base are far lighter than E-VQA's (5.6GB index, 2M KB, plus iNaturalist/Google Landmarks image sets — 50GB+ total).
- OMGM's reranker was trained on E-VQA but validated cross-dataset on InfoSeek by the original authors — so using their checkpoint on InfoSeek-only is already a tested setting, not a stretch.
- We use the **full released InfoSeek knowledge base** (100K entities) for retrieval — we do NOT shrink the KB. We only subset the number of test *queries* we evaluate on, not the retrieval difficulty itself. This keeps the comparison honest and realistic.

---

## 6. Data Volume Plan

```
Knowledge base:        full 100K entities (use as-is, no subsetting)
Entity/query images:   downloaded selectively (two-pass, see §7)

Query subset staging:
  Phase 0 (pilot/debug):      50 samples
  Phase 1 (failure analysis
           + main experiments): 500–1000 samples
  Phase 2 (optional, if time
           allows, for tighter
           confidence intervals): 2000+ samples
```

---

## 7. Data & Checkpoint Acquisition Plan

| Item | Status | Source |
|---|---|---|
| InfoSeek test.csv | **Needed** | OMGM repo (link works) |
| InfoSeek train.csv | Not needed | — |
| InfoSeek 100K Knowledge Base JSON | **Needed** | ⚠️ OMGM's Google Drive link is dead → use **HuggingFace mirror**: `BByrneLab/Infoseek-EchoSight-100K-KB` (file `infoseek_100k_wiki.zip`) |
| InfoSeek Summary FAISS Index (287MB) | **Needed** | OMGM repo (link works) |
| Wiki Images CSVs (269MB, metadata) | **Needed** | OMGM repo (link works) |
| Wiki entity images | **Needed, selective only** | Two-pass download (below) |
| Multimodal Fusion Reranker weights (2.3GB) | **Needed** | OMGM repo (link works) |
| Query images (Oven) | **Needed, selective only** | Only for our ~500-1000 sampled query IDs |
| Fine-tuned LLaVA-1.5-7B generator (12GB) | **Not needed** | Skip — it's E-VQA-specific; use a pretrained zero-shot LLM/MLLM instead (see below) |
| E-VQA everything (KB, FAISS, images) | **Not needed** | Skipping E-VQA entirely |

**Two-pass selective image download** (avoids downloading all 100K entity images upfront):
```
1. Run Step 1 FAISS search for our ~500-1000 queries first
   (needs only the 287MB index — no images required yet)
2. Collect the UNIQUE entity IDs appearing across all
   resulting top-20 candidate lists (expect ~5-15K unique
   entities, not 100K)
3. Filter the Wiki Images CSVs down to just those IDs
4. Download only those images
```

**Generator model:** use a pretrained, zero-shot LLM/MLLM (e.g., LLaMA-3-8B-Instruct in 4-bit, or InternVL-2.5-8B) downloaded fresh from HuggingFace — not OMGM's fine-tuned checkpoint. The paper's own Table 10 confirms zero-shot models work reasonably well with correct retrieved context, and our contribution is about retrieval control flow, not generator quality. Keep the **same generator** across baseline and our agent for a fair comparison.

**Estimated total storage:** ~10-12GB, comfortably within Kaggle's free-tier dataset limits.

**Sanity checks before proceeding to experiments:**
```
├── Confirm entity IDs match across: FAISS index ↔
│   KB JSON ↔ Wiki Images CSVs (research repos sometimes
│   have ID mismatches between release artifacts)
└── Confirm the KB JSON loads correctly and has ~100K
    entity keys
```

---

## 8. Execution Phases

### Phase 0 — Environment + Baseline Reproduction (Week 1)
- Set up Kaggle environment (conda env per OMGM's requirements; skip `flash-attn` if it's problematic on T4, fall back to standard attention)
- Download/verify all checkpoints and data from §7
- Run OMGM's released pipeline as-is on a 50-sample pilot subset
- Confirm the pipeline runs end-to-end and produces sane outputs
- **Deliverable:** working baseline inference pipeline

### Phase 1 — Failure Analysis (Week 1-2)
- Scale baseline run to 500-1000 query subset
- Collect all incorrect answers
- Categorize into Categories A-D (§4)
- Produce the failure breakdown table
- **Deliverable:** empirical motivation section for the paper; confirms (or revises) which agent actions to prioritize

### Phase 2 — Build Agent Tools & Controller (Week 2-4)
- Wrap each OMGM component as an independent callable tool
- Implement `tool_coarse_search_fused` (reuse Q-Former for search, new usage)
- Implement `tool_verify` (MLLM-based entity verification prompt)
- Implement `tool_verify_top2_full` (run Step 3 for top-2 candidates, compare)
- Implement the rule-based controller (confidence thresholds, branching, retry logic)
- Tune thresholds (τ1 for Step 1 ambiguity, τ2 for Step 2 ambiguity) via small grid search on a held-out dev split
- **Deliverable:** working agent pipeline, runnable end-to-end on the same query subset

### Phase 3 — Main Experiments (Week 4-5)
Run on the same 500-1000 sample subset, strictly same KB, same generator, for fair comparison:
```
├── Baseline: OMGM pipeline as released
├── Ours — verification only
├── Ours — fused search only
├── Ours — full agent (all actions enabled)
```
Report for each:
- Retrieval Recall@1 (entity-level) and section-level recall
- Final VQA accuracy (InfoSeek metrics: VQA accuracy for string answers, relaxed accuracy for numeric)
- **Average number of pipeline operations per query** (efficiency — report the honest per-query average, not worst-case, since extra actions only trigger on ambiguous cases)
- Error recovery: % of Category A/B/C/D failures (from Phase 1) that the agent fixes

### Phase 4 — Analysis & Ablations (Week 5-6)
- When does the agent retry/verify? (which question types, which failure patterns) — qualitative + quantitative breakdown
- Threshold sensitivity (how results change with τ1, τ2)
- Case studies: examples where agent recovers an error vs where it doesn't help
- Honest reporting of any downsides (extra latency, cases where verification itself errs)

### Phase 5 — Writing (Week 6-8)
- Paper structure (see §10 below)
- Target venues: ECIR, COLING, EMNLP Findings, or similar — not necessarily top-tier, per original constraint

---

## 9. Evaluation Metrics

```
Retrieval:
├── Recall@1 (entity-level, Step 1→2 output)
├── Section Recall@1 (Step 3 output, if we can establish
│   evidence-section ground truth from the KB)

VQA:
├── InfoSeek VQA Accuracy (string answers)
├── InfoSeek Relaxed Accuracy (numeric answers)

Efficiency:
├── Average # of tool calls per query (baseline vs agent)
├── Average wall-clock time per query

Our specific contribution metric:
├── % of Category A/B/C errors (from Phase 1 taxonomy)
    recovered by the agent, broken down by category
```

---

## 10. Target Paper Structure

```
1. Introduction
   - KB-VQA problem, OMGM's fixed-pipeline limitation
2. Related Work
   - OMGM, EchoSight, agentic RAG (text-domain)
3. Motivating Analysis  ← Phase 1 output goes here
   - Failure taxonomy on OMGM's own pipeline
4. Method
   - Agent tools (reused OMGM components + new fused-search,
     verify, verify-top2 actions)
   - Rule-based controller design
5. Experiments
   - Setup (dataset subset, honestly disclosed; see note below)
   - Main results table
   - Efficiency analysis
   - Error-category recovery analysis
6. Analysis / Discussion
   - When does the agent help vs not
   - Limitations (rule-based thresholds, subset size,
     InfoSeek-only, no learned policy)
7. Conclusion
```

**Note on subset disclosure (already agreed):** explicitly state that due to compute constraints we evaluate on a fixed subset of N queries (same subset for baseline and our method), re-running OMGM's released checkpoints ourselves on this subset to establish a fair, internally-consistent baseline — rather than directly citing the original paper's full-scale numbers.

---

## 11. Key Risks & Mitigations

```
Risk: Entity ID mismatches between FAISS index / KB JSON /
      image CSVs
→ Mitigate: verify with a small sanity check before Phase 1

Risk: flash-attn install issues on Kaggle T4
→ Mitigate: fall back to standard attention, slower but works

Risk: Phase 1 failure analysis shows Category D dominates
      (generator reasoning failures, not retrieval failures)
→ Mitigate: this is still a valid, honest finding — pivot
  paper framing to "agentic retrieval has limited headroom
  when X% of errors are generator-side, not retrieval-side"
  rather than forcing a method that won't show gains

Risk: Verification tool (MLLM-based) itself introduces errors
→ Mitigate: report this explicitly as a limitation, measure
  verification's own false-positive/negative rate

Risk: Kaggle session time limits (~30hrs/week) mid-experiment
→ Mitigate: checkpoint intermediate results (retrieval outputs,
  not just final answers) to disk/Kaggle datasets so runs
  can resume without redoing earlier pipeline stages
```

---

## 12. What's Already Decided (So We Don't Re-litigate)

- Dataset: InfoSeek only, full KB, subset of queries (500-1000 for main results)
- No retraining of any OMGM component — all frozen, inference-only
- No LLM-as-controller, no trained/RL policy — rule-based thresholds only
- Generator: pretrained zero-shot model from HuggingFace, not OMGM's fine-tuned checkpoint
- InfoSeek 100K KB source: HuggingFace mirror (`BByrneLab/Infoseek-EchoSight-100K-KB`), not OMGM's dead Google Drive link
- Phase 0 failure analysis happens **before** finalizing which agent actions to build, not after




---

## Where LLMs Could Appear (Three Separate Roles)

```
Role 1: THE CONTROLLER (decides which action to take next)
Role 2: THE VERIFIER (checks if entity matches image)
Role 3: THE GENERATOR (produces the final answer)
```

We already made a decision in the plan: **no LLM for Role 1.** Let me re-confirm why, then address Roles 2 and 3, where Qwen actually becomes relevant.

---

## Role 1: Controller — Confirmed, NOT an LLM

```
This was already settled in Phase 2 of the plan:

├── Controller = threshold-based rules
│   (score gaps between top-1/top-2 candidates)
├── NOT Qwen, NOT any LLM making "what should I do
│   next" decisions
└── Why: an LLM call per routing decision = 3-5+ extra
    inference calls PER QUERY just for control flow,
    which is slow and adds nothing scientifically
    (the novelty is in the control-flow STRUCTURE,
    not in how smart the router "thinks")

If you used Qwen here, you'd be reintroducing
exactly the "LLM-as-controller is expensive and
hard to isolate as a contribution" problem your
friend's note (and I) flagged earlier.
```

**So: no, we are not using Qwen 7B/14B as a "thinker" for the agent loop itself.** That part stays simple arithmetic (confidence gap > threshold → trigger action).

---

## Role 2: Verifier — This IS an MLLM, and Qwen is a Reasonable Choice

This is where your Qwen question actually applies.

```
tool_verify(query_image, candidate_entity_image)
    → needs to answer: "do these show the same
      entity/species/thing?"
    → this is a VISION-LANGUAGE task, needs an MLLM
      (not a text-only LLM — it has to actually
      look at two images)

Qwen 7B Instruct (text-only) → NOT sufficient here
Qwen-VL / Qwen2.5-VL (7B)    → sufficient, this is
                                 the right pick
```

```
Candidates for the verifier role:
├── Qwen2.5-VL-7B-Instruct      ← good choice, strong
│                                  on visual grounding tasks
├── LLaVA-1.5-7B / LLaVA-1.6    ← what OMGM itself uses,
│                                  easy comparability
├── InternVL-2.5-8B             ← also tested in OMGM's
│                                  own Table 10
```

**My recommendation: use the same model for Role 2 and Role 3** (verifier and generator), not two different models. This matters for your paper's clean-comparison argument — you don't want a reviewer asking "is the gain from your agent, or from using a better/different model for verification than generation?"

---

## Role 3: Generator — Also an MLLM Call, Same Model as Verifier

```
tool_generate(image, retrieved_section, question)
    → also needs to see the image + read text
    → also an MLLM task
```

**So Roles 2 and 3 should share one model.** This simplifies everything:

```
ONE MLLM used for both verify() and generate():
├── Qwen2.5-VL-7B-Instruct, OR
├── LLaVA-1.5-7B-Instruct (matches OMGM's own choice,
│   makes your efficiency/accuracy comparisons most
│   directly comparable to their reported numbers)
```

---

## Is Qwen2.5-VL-7B a Good Pick for Kaggle T4?

Yes — and your memory numbers for the Unsloth version are consistent with what I'd expect.

```
Qwen2.5-VL-7B-Instruct (Unsloth 4-bit):
├── ~5-6GB VRAM for the model itself
├── Fits comfortably in one T4 (16GB)
├── Leaves headroom for:
│   ├── CLIP (small, ~1-2GB)
│   ├── Q-Former reranker (modest)
│   └── FAISS (CPU-only, no VRAM)
└── With device_map="auto" across 2×T4 if needed,
    even more headroom
```

**Compared to LLaVA-1.5-7B** (what OMGM actually used for their generator):

```
                    Qwen2.5-VL-7B         LLaVA-1.5-7B
Vision encoder:     newer (dynamic res)    CLIP ViT-L/14 (fixed res)
Visual grounding:   generally stronger     decent, older
Instruction follow: strong                 decent
Unsloth 4-bit:       ~5GB                   should be similarly light
Directly comparable
to OMGM's Table 10:  no (different model)   yes (same model family
                                              they tested zero-shot)
```

---

##  Recommendation

```
Use LLaVA-1.5-7B (or LLaVA-1.6-7B), NOT Qwen2.5-VL,
for both verify() and generate() — specifically because
OMGM's own paper already reports zero-shot LLaVA-1.5-7B
numbers (Table 10: 17.2 E-VQA / 9.3 InfoSeek without
their retrieval, 41.81 / 33.29 with it).

Why this beats using Qwen:
├── You inherit a directly comparable reference point
│   from the original paper — "zero-shot LLaVA-1.5-7B
│   + OMGM retrieval" is already a documented number
├── Any improvement you show is cleanly attributable
│   to YOUR agent, not to "we just used a newer/better
│   base model than they did"
└── Reviewers will specifically appreciate that you
    didn't quietly swap in a stronger generator to
    inflate numbers
```

**That said, if you want to use Qwen2.5-VL-7B instead** (e.g., because the Unsloth 4-bit build is notably more memory-efficient and faster for you in practice on Kaggle, which is a legitimate practical reason), that's fine too — just:

```
├── Use it consistently for BOTH baseline (reproducing
│   OMGM's pipeline) AND your agent — never mix models
│   between the two
├── Note in the paper: "we substitute Qwen2.5-VL-7B
│   for LLaVA-1.5-7B as the generator for compute
│   efficiency reasons; both baseline and agent use
│   the identical generator for fair comparison"
└── This is a disclosed, defensible deviation —
    just don't silently compare your Qwen-powered
    agent against OMGM's paper-reported LLaVA numbers
```

---

## Updated Summary Table for the Plan

| Role | Model | Needs vision? | Notes |
|---|---|---|---|
| Controller (action routing) | **None — rule-based thresholds** | No | Confirmed, not an LLM |
| Verifier (`tool_verify`) | LLaVA-1.5-7B *(or Qwen2.5-VL-7B if swapping)* | **Yes** | Must be the VL/MLLM variant |
| Generator (`tool_generate`) | **Same model as Verifier** | Yes | Keep identical across baseline + agent |
