# FEVER Literature vs Our Results

This note identifies the source paper for the dataset we use, how much data **they** used vs **we** used, how our task differs from the official FEVER setup, and how our results compare to published numbers.

---

## 1. Source paper (dataset we used)

| | |
|---|---|
| **Title** | *FEVER: a Large-scale Dataset for Fact Extraction and VERification* |
| **Authors** | James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Arpit Mittal |
| **Venue** | **NAACL-HLT 2018** |
| **arXiv** | [1803.05355](https://arxiv.org/abs/1803.05355) |
| **ACL Anthology** | [N18-1074](https://aclanthology.org/N18-1074/) (PDF: [N18-1074.pdf](https://aclanthology.org/N18-1074.pdf)) |
| **Project site** | [fever.ai/dataset](https://fever.ai/dataset/fever.html) |

**What FEVER is:** Wikipedia-derived claims labeled **Supported**, **Refuted**, or **NotEnoughInfo (NEI)**, with sentence-level evidence for Supported/Refuted.

Our file: `/home/luk_viper/ML-Project/training_data.csv`  
Columns: `id`, `verifiable`, `claim`, `evidence`.

### Label mapping (from exact class counts)

| Our `verifiable` value | Count in CSV | Matches FEVER train |
|---|---:|---|
| `factual` | **109,810** | Supported (80,035) + Refuted (29,775) |
| `non-factual` | **35,639** | NotEnoughInfo (35,639) |

So in this project, binary classification is:

- **factual** = claim is *verifiable* as Supported **or** Refuted (evidence exists in FEVER)  
- **non-factual** = **NotEnoughInfo**

We train **claim-only** models (the `evidence` column is **not** used as input).

---

## 2. How much data **they** used (Thorne et al. 2018)

**Total FEVER corpus: 185,445 claims.**

Official split (Table 1 in the paper):

| Split | Supported | Refuted | NotEnoughInfo (NEI) | **Total** |
|---|---:|---:|---:|---:|
| **Training** | 80,035 | 29,775 | 35,639 | **145,449** |
| **Dev** | 3,333 | 3,333 | 3,333 | **9,999** |
| **Test** | 3,333 | 3,333 | 3,333 | **9,999** |
| **Reserved** (shared-task test) | 6,666 | 6,666 | 6,666 | **19,998** |
| **All** | | | | **185,445** |

Notes from the paper:

- Each Wikipedia page used to generate claims appears in **exactly one** split.  
- Dev/test are **balanced** across the 3 classes.  
- Training is **imbalanced** (far more Supported than Refuted/NEI).

---

## 3. How much data **we** used

### 3.1 Source file

| Stage | Amount |
|---|---:|
| Rows in `training_data.csv` | **145,449** (= FEVER **training** split) |
| After `drop_duplicates(subset="claim")` | **135,545** |
| `factual` in CSV | **109,810** |
| `non-factual` in CSV | **35,639** |

We do **not** use the paper’s separate 9,999-dev / 9,999-test / 19,998-reserved files; we resample from the train CSV and make our own 80/20 holdout.

### 3.2 Balanced experiment sets

Equal factual / non-factual (50/50), then **80% development / 20% holdout** (stratified).

| Setting | Total claims | Per class | Development (80%) | Holdout (20%) |
|---|---:|---:|---:|---:|
| **4k** | **4,000** | 2,000 + 2,000 | **3,200** | **800** |
| **67k** (full balanced) | **67,134** | 33,567 + 33,567 | **53,707** | **13,427** |

### 3.3 Side-by-side: paper vs us

| | Thorne et al. | Our project |
|---|---|---|
| Claims available | **185,445** (all splits) | **145,449** train file → **135,545** unique claims |
| Labels used | 3-way SUP / REF / NEI | 2-way factual (SUP∪REF) vs non-factual (NEI) |
| Typical train size in paper baseline | Full train **145,449** (3-class, + evidence pipeline) | **4,000** or **67,134** balanced claim-only |
| Evaluation split | Fixed FEVER test (9,999) | Random stratified **20% holdout** of our balanced sample |

```text
Data volume (claims)

FEVER full corpus          ████████████████████  185,445
FEVER train (our CSV)      ███████████████░░░░░  145,449
Our unique claims          ██████████████░░░░░░  135,545
Our max balanced (67k)     ███████░░░░░░░░░░░░░   67,134
Our small balanced (4k)    ░░░░░░░░░░░░░░░░░░░░    4,000
```

---

## 4. Related paper most comparable to *our* setup

Because we classify **claims without evidence**, the closest published analysis is:

| | |
|---|---|
| **Title** | *Towards Debiasing Fact Verification Models* |
| **Authors** | Tal Schuster, Darsh J. Shah, Yun Jie Serene Yeo, Daniel Filizzola, Enrico Santus, Regina Barzilay |
| **Venue** | **EMNLP 2019** |
| **ACL** | [D19-1341](https://aclanthology.org/D19-1341/) |

They show a **claim-only BERT** can score surprisingly high on FEVER *without reading evidence*, largely due to claim wording cues.

---

## 5. Important fairness warning

| Aspect | Official FEVER (Thorne et al.) | Our project |
|---|---|---|
| Labels | 3-way: Supported / Refuted / NEI | **2-way:** factual (SUP∪REF) / non-factual (NEI) |
| Evidence | Required for full FEVER score | **Not used** (claim text only) |
| Data volume | Train 145k + fixed test 10k | Balanced **4k** or **67k** + 20% holdout |
| Balance | Dev/test balanced 3-way; train skewed | We **force 50/50** binary |
| Metric | Label accuracy + **FEVER score** (label ∧ correct evidence) | Holdout / test **accuracy, F1, AUC** |
| Model | Retrieval + RTE pipeline | Frozen embeddings + classical / stacking classifiers |

We **cannot** claim to beat the official FEVER leaderboard. We compare against:

1. Thorne pipeline **label accuracy ignoring evidence**, and oracle RTE numbers (context).  
2. Schuster **claim-only** accuracies (closest spirit).  
3. Chance on a **balanced binary** task (~50%).

---

## 6. Published numbers (literature)

### 6.1 Thorne et al. (NAACL 2018) — full pipeline & oracles

| Setting | Accuracy | Notes |
|---|---:|---|
| Full pipeline, **correct evidence required** (test) | **31.87%** | Primary hard metric |
| Full pipeline, **ignore evidence correctness** (test) | **50.91%** | Label-only accuracy of pipeline |
| Dev pipeline DA (poster / paper variants) | ~52% | Label accuracy of full system |
| Oracle RTE (DA) with **gold evidence** (dev) | **80.82%** (NearestP) / **88.00%** (RandomS) | Upper bound if evidence is perfect |
| Random 3-class (balanced) | ~33.3% | Chance |

Pipeline: TF-IDF document retrieval (DrQA) → sentence selection → Decomposable Attention RTE.  
Trained/evaluated on the **full FEVER splits** above (train ~145k examples for the pipeline components).

### 6.2 Schuster et al. (EMNLP 2019) — claim-only vs evidence-aware (3-class FEVER)

| Model | Uses evidence? | Accuracy (3-class) |
|---|---|---:|
| Majority baseline | — | 33.3% |
| DA (Thorne-style evidence-aware) | Yes | 52.1% |
| NSMN (strong evidence-aware) | Yes | 69.7% |
| InferSent claim-only (random emb.) | No | 54.1% |
| InferSent claim-only (GloVe) | No | 57.3% |
| **BERT claim-only** | **No** | **61.7%** |

---

## 7. Our best results (for comparison)

Sources: `results_easy/`, `results_stacking_17/MASTER_stacking_17.xlsx`, `results_cross_fold/`.

### 7.1 Claim-only binary, balanced, holdout/test

| Pipeline | Size used | Embedder | Best accuracy | Notes |
|---|---|---|---:|---|
| `fever_easy` Stacking (4 bases → LR meta) | **67,134** | BERT/T5* | **0.654** | Default mode, 80/20 test |
| `fever_easy` Stacking | **67,134** | MiniLM | 0.627 | |
| `fever_easy` Stacking | **67,134** | TF-IDF@500 | 0.594 | |
| **17-base normal stacking** (top meta) | **67,134** | BERT/T5* | **0.654** | Shared OOF + top meta |
| **17-base cross-fold stacking** | **67,134** | BERT/T5* | **0.653** | Best E* ≈ E2 |
| 17-base cross-fold | **4,000** | BERT/T5* | **0.638** | E2, LDA meta |
| Single-base cross-fold (earlier) | **67,134** | BERT/T5* | **0.645** | LogReg base → LogReg meta |

\*In this environment BERT/T5 slots often used **DistilBERT mean-pool** fallback when HF download was blocked.

Chance on our **balanced binary** task ≈ **50%**.

### 7.2 Side-by-side with literature (conceptual)

| Reference | Data / task | Acc | vs our best (~65% on 67k binary) |
|---|---|---:|---|
| Chance (our binary balanced) | 4k or 67k, 2-class | 50% | We are **~15 pts above chance** |
| Thorne pipeline (ignore evidence) | Full FEVER test (~10k), 3-class + retrieval | 50.9% | Different task & data size |
| Thorne full FEVER score | Full FEVER test, 3-class + evidence | 31.9% | Not comparable (we don’t retrieve evidence) |
| Schuster claim-only BERT | FEVER 3-class, claim only | **61.7%** | Closest claim-only published number; our binary ~65% is **same ballpark** |
| Schuster evidence-aware NSMN | 3-class + evidence | 69.7% | Higher; they use evidence |
| Thorne oracle RTE (gold evidence) | 3-class | 81–88% | Upper bound with perfect evidence |

```text
Accuracy (different setups — compare with care)

Chance (binary balanced)     ████████████░░░░░░░░  50%
Thorne pipeline (3-class)*   ████████████░░░░░░░░  51%   *ignore evidence
Schuster claim-only BERT     ███████████████░░░░░  62%   3-class
Our best claim-only stack    ████████████████░░░░  65%   2-class, 67k balanced
Schuster NSMN + evidence     █████████████████░░░  70%
Thorne oracle + gold evid.   ████████████████████  81–88%
```

---

## 8. How to read the comparison (viva / report)

1. **Cite Thorne et al. (2018)** as the dataset paper; total FEVER = **185,445** claims; we start from their **train** split (**145,449**).  
2. **State data we used:** balanced **4,000** or **67,134** claims (50/50), with **80/20** holdout (3,200/800 or 53,707/13,427).  
3. **State the binary labels:** factual ≈ Supported∪Refuted, non-factual ≈ NEI (from CSV counts).  
4. **State clearly** that we study **claim-only** classification — FEVER score (**31.87%**) is **not** our target metric.  
5. **Cite Schuster et al. (2019)** for claim-only FEVER behaviour; our ~65% on 67k is **above chance** and in a similar regime to their claim-only BERT (61.7% on 3-class).  
6. **Do not claim SOTA on FEVER.** Evidence-aware systems use Wikipedia evidence we omit.  
7. Our contribution is methodological: **multi-classifier stacking** and **cross-fold data-exchange stacking** on balanced FEVER claims.

**Faculty one-liner:** *We use Thorne et al.’s FEVER train split (145,449 claims), then run on a balanced subset (4k or ~67k). Our claim-only binary accuracy (~65% on 67k) is above chance and comparable in spirit to published claim-only FEVER results, while using far less data than the full 185k corpus and no evidence retrieval.*

---

## 9. References (BibTeX-ready)

```bibtex
@inproceedings{thorne-etal-2018-fever,
  title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification},
  author = {Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
  booktitle = {NAACL-HLT},
  year = {2018},
  url = {https://aclanthology.org/N18-1074/}
}

@inproceedings{schuster-etal-2019-towards,
  title = {Towards Debiasing Fact Verification Models},
  author = {Schuster, Tal and Shah, Darsh J. and Yeo, Yun Jie Serene and Filizzola, Daniel and Santus, Enrico and Barzilay, Regina},
  booktitle = {EMNLP-IJCNLP},
  year = {2019},
  url = {https://aclanthology.org/D19-1341/}
}
```

Optional shared-task overview: Thorne et al., *The Fact Extraction and VERification (FEVER) Shared Task*, 2018 ([W18-5501](https://aclanthology.org/W18-5501/)).
