# Cross-Fold Data-Exchange Stacking

**Faculty demo script:** [`cross_fold_easy.py`](cross_fold_easy.py)  
**Full experiment package:** [`cross_fold_stacking/`](cross_fold_stacking/)  
**Results Excel:** [`results_cross_fold/MASTER_cross_fold_stacking.xlsx`](results_cross_fold/MASTER_cross_fold_stacking.xlsx)

---

## 1. Idea in one sentence

We train a **Base model** and a **Meta model** on *overlapping but not identical* parts of the data, then deliberately **swap a few samples** between the Base’s test fold and the Meta’s test fold, and stack Base probability scores into the Meta model — without leaking labels through in-sample predictions.

---

## 2. Big picture

```text
                    FULL BALANCED FEVER DATA
                              │
              ┌───────────────┴───────────────┐
              │ 80% DEVELOPMENT               │ 20% HOLD OUT
              │ (folds + exchange + train)    │ (NEVER used in training)
              └───────────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           F1 F2 F3        F4 F5         (stratified)
              │               │
              └───────┬───────┘
                      │
         optional sample exchange A ↔ B
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
   BASE MODEL                  META MODEL
   (Logistic Reg.)             (Logistic Reg.)
   learns from text            learns from Base
   features                    probability scores
         │                         │
         └──────────┬──────────────┘
                    ▼
            FINAL PREDICTION
         (factual / non-factual)
```

---

## 3. How the data is split

### Step A — Hold out 20%

| Split | Size | Used for training? |
|---|---|---|
| Development | 80% | Yes (folds / exchange / Base / Meta) |
| Hold out | 20% | **No** — final report only |

### Step B — Five equal folds on development

Each of **F1 … F5** is ~20% of development (stratified by label).

```text
 DEVELOPMENT (80%)
 ┌──────┬──────┬──────┬──────┬──────┐
 │  F1  │  F2  │  F3  │  F4  │  F5  │
 │ ~20% │ ~20% │ ~20% │ ~20% │ ~20% │
 └──────┴──────┴──────┴──────┴──────┘
```

---

## 4. Base model vs Meta model — who sees what?

### 4.1 Initial allocation (before any exchange)

| Fold | Base Model | Meta Model | Meaning |
|---|---|---|---|
| **F1** | Training | **Testing** | Meta’s private test fold |
| **F2** | Training | Training | Shared train |
| **F3** | Training | Training | Shared train |
| **F4** | Training | Training | Shared train |
| **F5** | **Testing** | Training | Base’s private test fold |

```text
                 BASE                         META
        ┌─────────────────────┐      ┌─────────────────────┐
  TRAIN │ F1  F2  F3  F4      │      │     F2  F3  F4  F5  │
        └─────────────────────┘      └─────────────────────┘
  TEST  │              F5     │      │ F1                  │
        └─────────────────────┘      └─────────────────────┘

  Shared training region = F2 ∪ F3 ∪ F4
  (both models train here → Base scores for Meta must be out-of-fold)
```

### 4.2 Cross-fold sample exchange (the research idea)

We do **not** swap whole folds.  
We pick equal-sized subsets:

- **A** ⊂ F1  
- **B** ⊂ F5  
- |A| = |B|  
- Exchange ratio experiments: **0% / 10% / 25% / 40%** (E0–E3)

```text
   F1:  [........ A ........]          F5:  [........ B ........]
              │                                    │
              └──────────── swap roles ────────────┘
```

### 4.3 Roles AFTER exchange

| Data piece | Base Model | Meta Model |
|---|---|---|
| F1 **without** A | Training | Testing |
| **A** (from F1) | **Testing** | **Training** |
| F2 | Training | Training |
| F3 | Training | Training |
| F4 | Training | Training |
| F5 **without** B | Testing | Training |
| **B** (from F5) | **Training** | **Testing** |

```mermaid
flowchart TB
  subgraph DEV["Development 80%"]
    F1a["F1 without A"]
    A["A ⊂ F1"]
    F2["F2"]
    F3["F3"]
    F4["F4"]
    F5b["F5 without B"]
    B["B ⊂ F5"]
  end

  subgraph BASE["Base Model"]
    BT["TRAIN"]
    BTe["TEST"]
  end

  subgraph META["Meta Model"]
    MT["TRAIN"]
    MTe["TEST"]
  end

  F1a --> BT
  F1a --> MTe
  A --> BTe
  A --> MT
  F2 --> BT
  F2 --> MT
  F3 --> BT
  F3 --> MT
  F4 --> BT
  F4 --> MT
  F5b --> BTe
  F5b --> MT
  B --> BT
  B --> MTe
```

**ASCII version of the same idea:**

```text
 AFTER EXCHANGE

 BASE TRAIN  = (F1\A) + F2 + F3 + F4 + B
 BASE TEST   = A + (F5\B)

 META TRAIN  = A + F2 + F3 + F4 + (F5\B)
 META TEST   = (F1\A) + B
```

---

## 5. How stacking is done

### 5.1 Two-level models

| Level | Model | Input | Output |
|---|---|---|---|
| **Base** | Logistic Regression | claim features (TF-IDF@500 / MiniLM / BERT / RoBERTa / T5) | probability of *non-factual* |
| **Meta** | Logistic Regression | **only** that Base probability (1 number) | final factual / non-factual |

### 5.2 Flow

```mermaid
flowchart LR
  C["Claim text"] --> FEAT["Embedder\nTF-IDF / MiniLM / BERT / …"]
  FEAT --> BASE["Base LogReg"]
  BASE -->|"P(non-factual)\nOOF if needed"| META["Meta LogReg"]
  META --> Y["Final label"]
```

```text
  claim  →  features X  →  Base  →  score s = P(non-factual)
                                      │
                                      ▼
                                   Meta(s)  →  ŷ
```

### 5.3 Why out-of-fold (OOF) scores?

F2, F3, F4 are **training data for both** Base and Meta.

If Meta trained on Base scores computed *while Base had already seen those same rows*, Meta would get an unrealistically easy signal → **leakage**.

**Rule we enforce:**

```text
  For every sample used to train Meta:
    if Base trained on it  →  use a Base clone that held it out (inner CV)
    if Base did NOT train on it  →  use the full Base model (already OOF)
```

```text
  Meta TRAIN samples
  ├── A and (F5\B)     →  in Base TEST  →  full Base score is safe
  └── F2 ∪ F3 ∪ F4     →  in Base TRAIN →  inner 5-fold OOF scores
```

### 5.4 Final holdout evaluation

```text
  1. Fit Base on Base-TRAIN (development only)
  2. Fit Meta on OOF Base scores of Meta-TRAIN
  3. Holdout:
        s_hold = Base.predict_proba(X_holdout)
        ŷ_hold = Meta.predict(s_hold)
  4. Holdout was never in any train / tune / exchange step
```

---

## 6. Experiments E0–E3

| Code | Exchange ratio | What changes |
|---|---:|---|
| **E0** | 0% | Baseline — no swap (A=B=empty) |
| **E1** | 10% | Small controlled overlap change |
| **E2** | 25% | Medium exchange |
| **E3** | 40% | Larger exchange |

For each experiment we report:

- Base test accuracy  
- Meta / stacking accuracy (on Meta’s test roles)  
- Precision, Recall, F1, ROC-AUC  
- **Holdout stacking** performance (main comparison)  
- Train/test counts + sample allocation manifest  

---

## 7. How to run (faculty demo)

```bash
cd /home/luk_viper/ML
source .venv/bin/activate

# One file demo — E0–E3 on 4k TF-IDF (good for viva)
python cross_fold_easy.py --size 4k --embedder tfidf --all-ratios

# Other embedders / full balanced data
python cross_fold_easy.py --size 4k  --embedder minilm --all-ratios
python cross_fold_easy.py --size 67k --embedder bert   --all-ratios
```

Demo Excel is written to:

`results_cross_fold/faculty_demo/faculty_demo_<size>_<embedder>.xlsx`

Sheets:

| Sheet | Contents |
|---|---|
| `Summary` | All metrics for Base / Meta / Holdout |
| `Holdout_Stacking` | E0–E3 comparison on untouched 20% |
| `Diagram` | Text diagram of roles + stacking |
| `Manifest_E*` | Which fold / role each sample had |

Full matrix already run (all sizes × embedders):

`results_cross_fold/MASTER_cross_fold_stacking.xlsx`

---

## 8. Leakage rules (checklist for viva)

1. Holdout 20% never enters training, exchange, or tuning.  
2. No sample is both train **and** test for the **same** model.  
3. Meta training features are Base scores that are **out-of-fold**.  
4. Final numbers use predictions on samples neither model trained on (holdout).  
5. Cross-model training exposure (F2∪F3∪F4) is reported and handled with OOF.

---

## 9. Note on embedders in this environment

If HuggingFace download is blocked, **BERT** / **T5** slots use a local **DistilBERT mean-pool** fallback, and **RoBERTa** uses **DistilBERT CLS-pool** (same approach as `fever_easy.py`). MiniLM can reuse the project embedding cache.
