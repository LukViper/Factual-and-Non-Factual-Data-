# Final Comparison: Normal Stacking vs Cross-Fold Stacking

**Results Excel:** [`results_stacking_17/MASTER_stacking_17.xlsx`](results_stacking_17/MASTER_stacking_17.xlsx)  
**Code:** [`stacking_17/`](stacking_17/)

This report compares the two stacking methods after using the project’s classifier suite as **bases** and the **top validation performers** as **meta** models.

---

## 1. What was compared

| | **Normal stacking** (“base”) | **Cross-fold stacking** |
|---|---|---|
| **Idea** | Classic multi-base stacker: OOF base scores → meta model | Same bases/meta, plus F1–F5 roles and F1↔F5 sample exchange (E0–E3) |
| **Bases** | Individual classifiers from the 17-suite (Voting & nested Stacking excluded) | Same |
| **Meta** | Top-3 by validation accuracy on development | Same ranking rule |
| **Split** | 80% development / 20% holdout | **Same** split & features |
| **Main metric** | Holdout accuracy / F1 / ROC-AUC | Holdout stacking accuracy / F1 / ROC-AUC |

**4k bases:** 15 classifiers (LR, Perceptron, LDA, KNN, DT, ET, RF, AdaBoost, GB/HistGB, SVM, NB, MLP, XGB, LightGBM, CatBoost).  
**67k bases:** same suite **minus KNN & MLP** (too slow on ~50k); Gradient Boosting uses `HistGradientBoosting`.

---

## 2. Head-to-head (best holdout per size × embedder)

Δ Acc = cross-fold best − normal best.  
Winner = cross-fold if Δ > 0.001, normal if Δ < −0.001, else tie.

| Size | Embedder | Normal best (meta) | Acc | F1 | Cross-fold best | Acc | F1 | Δ Acc | Winner |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| 4k | bert | Extra Trees | 0.6175 | 0.6117 | **E2** LDA | **0.6375** | 0.6338 | +0.0200 | **cross-fold** |
| 4k | t5 | Extra Trees | 0.6175 | 0.6117 | **E2** LDA | **0.6375** | 0.6338 | +0.0200 | **cross-fold** |
| 4k | minilm | Logistic Regression | 0.5888 | 0.5923 | **E2** LogReg | **0.6112** | 0.6088 | +0.0225 | **cross-fold** |
| 4k | roberta | Gradient Boosting | 0.6050 | 0.6099 | **E3** LDA | **0.6275** | 0.6237 | +0.0225 | **cross-fold** |
| 4k | tfidf | LightGBM | 0.5212 | 0.4845 | **E1** GradBoost | **0.5738** | 0.5297 | +0.0525 | **cross-fold** |
| 67k | bert | Logistic Regression | 0.6536 | 0.6610 | E2 LogReg | 0.6533 | 0.6607 | −0.0003 | tie |
| 67k | t5 | Logistic Regression | 0.6536 | 0.6610 | E2 LogReg | 0.6533 | 0.6607 | −0.0003 | tie |
| 67k | minilm | **LightGBM** | **0.6240** | 0.6377 | E0 SVM | 0.6220 | 0.6326 | −0.0019 | **normal** |
| 67k | roberta | LogReg | 0.6435 | 0.6543 | **E0** SVM | **0.6446** | 0.6556 | +0.0011 | **cross-fold** |
| 67k | tfidf | GradBoost | 0.6002 | 0.6192 | **E0** LightGBM | **0.6024** | 0.6309 | +0.0022 | **cross-fold** |

```text
Wins (holdout accuracy)
  Cross-fold : 7 / 10
  Normal     : 1 / 10
  Tie        : 2 / 10
```

---

## 3. Visual summary (holdout accuracy)

```text
4k
  tfidf    Normal ████████████████████░░░░  0.521
           Cross  ███████████████████████░  0.574   ← +5.3 pts (largest gain)

  minilm   Normal ███████████████████████░  0.589
           Cross  ████████████████████████  0.611   ← +2.3 pts

  roberta  Normal ████████████████████████  0.605
           Cross  █████████████████████████ 0.628   ← +2.3 pts

  bert/t5  Normal ████████████████████████  0.618
           Cross  █████████████████████████ 0.638   ← +2.0 pts

67k
  tfidf    Normal / Cross ≈ 0.600 / 0.602   (near tie)
  minilm   Normal slightly ahead            0.624 vs 0.622
  roberta  Cross slightly ahead             0.645 vs 0.643
  bert/t5  essentially tied                 ≈ 0.654
```

---

## 4. Cross-fold exchange effect (E0–E3)

Max holdout accuracy across metas for each exchange setting:

| Size | Embedder | E0 (0%) | E1 (10%) | E2 (25%) | E3 (40%) | Best |
|---|---|---:|---:|---:|---:|---|
| 4k | bert | 0.6338 | 0.6325 | **0.6375** | 0.6225 | E2 |
| 4k | t5 | 0.6338 | 0.6325 | **0.6375** | 0.6225 | E2 |
| 4k | minilm | 0.5975 | 0.6062 | **0.6112** | 0.5888 | E2 |
| 4k | roberta | 0.6262 | 0.6225 | 0.6262 | **0.6275** | E3 |
| 4k | tfidf | 0.5250 | **0.5738** | 0.5438 | 0.5238 | E1 |
| 67k | bert | 0.6532 | 0.6501 | **0.6533** | 0.6520 | E2 |
| 67k | t5 | 0.6532 | 0.6501 | **0.6533** | 0.6520 | E2 |
| 67k | minilm | **0.6220** | 0.6201 | 0.6217 | 0.6207 | E0 |
| 67k | roberta | **0.6446** | 0.6434 | 0.6416 | 0.6387 | E0 |
| 67k | tfidf | **0.6024** | 0.6021 | 0.5943 | 0.6007 | E0 |

**Pattern:** On **4k**, a non-zero exchange (often **E1/E2**) helps. On **67k**, **E0** (no exchange) is often best or tied — more data already stabilizes the stacker.

---

## 5. Method differences (why scores differ)

```text
NORMAL STACKING
  Development → OOF scores from all bases → Meta(top-K) → Holdout

CROSS-FOLD STACKING
  Development → F1…F5 roles (+ optional A↔B exchange)
             → OOF base scores under Base-train rules
             → Meta(top-K) on Meta-train features
             → Holdout
```

| Topic | Normal | Cross-fold |
|---|---|---|
| Data roles | One train pool for all bases | Base and Meta have **different** train/test folds |
| Exchange | None | Controlled F1↔F5 sample swap (0–40%) |
| Leakage control | Standard OOF for bases | OOF **plus** exchange allocation rules |
| Typical 4k outcome | Strong, but beaten by exchange | **Higher** holdout accuracy |
| Typical 67k outcome | Competitive / slightly better sometimes | Nearly the same as normal |

---

## 6. Takeaways for faculty / report

1. **With multi-base + top meta, cross-fold wins most head-to-heads (7/10)**, especially on **4k** (+2 to +5 accuracy points).
2. **On full balanced ~67k, the two methods converge** (differences ≈ 0–0.2 points). Exchange adds little once data is large.
3. **Best embedders** in both methods: **BERT / T5** (DistilBERT mean-pool fallback) ≈ **0.65** holdout on 67k; TF-IDF is weakest.
4. **Meta choice matters:** top validation models were often LDA / LogReg / SVM / LightGBM / Extra Trees / Gradient Boosting depending on size and embedder.
5. Use **cross-fold** when studying controlled train-distribution overlap; use **normal stacking** as the strong multi-base baseline.

**One-liner:** *Cross-fold data-exchange stacking improves holdout accuracy over normal multi-base stacking on the 4k balanced set; on 67k the two approaches perform almost the same.*

---

## 7. File map

| Path | What |
|---|---|
| [`results_stacking_17/MASTER_stacking_17.xlsx`](results_stacking_17/MASTER_stacking_17.xlsx) | All metrics, rankings, best compare |
| `results_stacking_17/<size>_<embedder>/combined_*.xlsx` | Per-cell normal + cross-fold |
| [`stacking_17/README.md`](stacking_17/README.md) | How to re-run |
| [`CROSS_FOLD_STACKING_README.md`](CROSS_FOLD_STACKING_README.md) | Earlier single-base cross-fold design + diagrams |
