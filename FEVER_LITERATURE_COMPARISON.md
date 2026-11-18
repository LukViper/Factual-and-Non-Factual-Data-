# FEVER Literature vs Our Results

This note identifies the source paper for the dataset we use, explains how our task differs from the official FEVER setup, and compares published numbers with our best runs.

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

**What FEVER is:** ~185k Wikipedia-derived claims labeled **Supported**, **Refuted**, or **NotEnoughInfo**, with sentence-level evidence for Supported/Refuted.

Our CSV (`training_data.csv`) matches that release: columns `id`, `verifiable`, `claim`, `evidence`. We map:

| Our label | FEVER label |
|---|---|
| `factual` | Supported |
| `non-factual` | Refuted |

We **drop NotEnoughInfo** and train **claim-only** classifiers (evidence is not used as model input). Counts in our file: ~109.8k factual + ~35.6k non-factual (then we balance for experiments).

---

## 2. Related paper most comparable to *our* setup

Because we classify **claims without evidence**, the closest published analysis is:

| | |
|---|---|
| **Title** | *Towards Debiasing Fact Verification Models* |
| **Authors** | Tal Schuster, Darsh J. Shah, Yun Jie Serene Yeo, Daniel Filizzola, Enrico Santus, Regina Barzilay |
| **Venue** | **EMNLP 2019** |
| **ACL** | [D19-1341](https://aclanthology.org/D19-1341/) |

They show a **claim-only BERT** can score surprisingly high on FEVER *without reading evidence*, largely due to claim wording cues (e.g. negation → Refuted).

---

## 3. Important fairness warning

| Aspect | Official FEVER (Thorne et al.) | Our project |
|---|---|---|
| Labels | 3-way: Supported / Refuted / NEI | **2-way:** factual / non-factual |
| Evidence | Required for full FEVER score | **Not used** (claim text only) |
| Balance | Shared-task splits are balanced 3-way | We **balance** factual vs non-factual |
| Metric | Label accuracy + **FEVER score** (label ∧ correct evidence) | Holdout / test **accuracy, F1, AUC** |
| Model | Retrieval + RTE pipeline | Frozen embeddings + classical / stacking classifiers |

So we **cannot** claim to beat the official FEVER leaderboard. We compare against:

1. Thorne pipeline **label accuracy ignoring evidence**, and oracle RTE numbers (context).  
2. Schuster **claim-only** accuracies (closest spirit).  
3. Chance / majority on a **balanced binary** task (~50%).

---

## 4. Published numbers (literature)

### 4.1 Thorne et al. (NAACL 2018) — full pipeline & oracles

| Setting | Accuracy | Notes |
|---|---:|---|
| Full pipeline, **correct evidence required** (test) | **31.87%** | Primary hard metric |
| Full pipeline, **ignore evidence correctness** (test) | **50.91%** | Label-only accuracy of pipeline |
| Dev pipeline DA (poster / paper variants) | ~52% | Label accuracy of full system |
| Oracle RTE (DA) with **gold evidence** (dev) | **80.82%** (NearestP) / **88.00%** (RandomS) | Upper bound if evidence is perfect |
| Random 3-class (balanced) | ~33.3% | Chance |

Pipeline sketch from the paper: TF-IDF document retrieval (DrQA) → sentence selection → Decomposable Attention RTE.

### 4.2 Schuster et al. (EMNLP 2019) — claim-only vs evidence-aware (3-class FEVER)

| Model | Uses evidence? | Accuracy (3-class) |
|---|---|---:|
| Majority baseline | — | 33.3% |
| DA (Thorne-style evidence-aware) | Yes | 52.1% |
| NSMN (strong evidence-aware) | Yes | 69.7% |
| InferSent claim-only (random emb.) | No | 54.1% |
| InferSent claim-only (GloVe) | No | 57.3% |
| **BERT claim-only** | **No** | **61.7%** |

Their point: claim-only BERT is **far above chance** on FEVER, so claim text alone carries strong label cues.

---

## 5. Our best results (for comparison)

Sources: `results_easy/`, `results_stacking_17/MASTER_stacking_17.xlsx`, `results_cross_fold/`.

### 5.1 Claim-only binary (factual vs non-factual), balanced, holdout/test

| Pipeline | Size | Embedder | Best accuracy | Notes |
|---|---|---|---:|---|
| `fever_easy` Stacking (4 bases → LR meta) | 67k | BERT/T5* | **0.654** | Default mode, 80/20 test |
| `fever_easy` Stacking | 67k | MiniLM | 0.627 | |
| `fever_easy` Stacking | 67k | TF-IDF@500 | 0.594 | |
| **17-base normal stacking** (top meta) | 67k | BERT/T5* | **0.654** | Shared OOF + top meta |
| **17-base cross-fold stacking** | 67k | BERT/T5* | **0.653** | Best E* ≈ E2 |
| 17-base cross-fold | 4k | BERT/T5* | **0.638** | E2, LDA meta |
| Single-base cross-fold (earlier) | 67k | BERT/T5* | **0.645** | LogReg base → LogReg meta |

\*In this environment BERT/T5 slots often used **DistilBERT mean-pool** fallback when HF download was blocked.

Chance on our **balanced binary** task ≈ **50%**.

### 5.2 Side-by-side with literature (conceptual)

| Reference | Task | Acc | vs our best (~65% binary claim-only) |
|---|---|---:|---|
| Chance (our binary balanced) | 2-class claim-only | 50% | We are **~15 pts above chance** |
| Thorne pipeline (ignore evidence) | 3-class + retrieval | 50.9% | Different task; both sit near “hard” FEVER label-only regime |
| Thorne full FEVER score | 3-class + evidence | 31.9% | Not comparable (we don’t retrieve evidence) |
| Schuster claim-only BERT | **3-class** claim-only | **61.7%** | Closest published claim-only number; our binary ~65% is **same ballpark**, not directly interchangeable |
| Schuster evidence-aware NSMN | 3-class + evidence | 69.7% | Higher; they use evidence |
| Thorne oracle RTE (gold evidence) | 3-class | 81–88% | Upper bound with perfect evidence — far above claim-only |

```text
Accuracy (different setups — compare with care)

Chance (binary balanced)     ████████████░░░░░░░░  50%
Thorne pipeline (3-class)*   ████████████░░░░░░░░  51%   *ignore evidence
Schuster claim-only BERT     ███████████████░░░░░  62%   3-class
Our best claim-only stack    ████████████████░░░░  65%   2-class balanced 67k
Schuster NSMN + evidence     █████████████████░░░  70%
Thorne oracle + gold evid.   ████████████████████  81–88%
```

---

## 6. How to read the comparison (viva / report)

1. **Cite Thorne et al. (2018)** as the dataset paper; state that our `factual` / `non-factual` labels are Supported / Refuted.
2. **State clearly** that we study **claim-only binary classification**, not the full FEVER evidence pipeline — so FEVER score (31.87%) is **not** our target metric.
3. **Cite Schuster et al. (2019)** to justify that claim-only models can exceed chance on FEVER because of claim-side cues; our ~65% binary accuracy is **consistent** with that finding (stronger than 50% chance; similar spirit to their 61.7% 3-class claim-only BERT).
4. **Do not claim SOTA on FEVER.** Evidence-aware systems (NSMN ~70% 3-class, later systems higher) use Wikipedia evidence we deliberately omit.
5. Our contribution is methodological: **multi-classifier stacking** and **cross-fold data-exchange stacking** on balanced FEVER claims — not a new FEVER retrieval architecture.

**Faculty one-liner:** *We use the FEVER dataset of Thorne et al. (NAACL 2018), restricted to Supported vs Refuted claims and claim text only. Our best balanced binary accuracy (~65%) sits above chance and in the same regime as published claim-only FEVER classifiers (Schuster et al., EMNLP 2019), while remaining below evidence-aware FEVER systems.*

---

## 7. References (BibTeX-ready)

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
