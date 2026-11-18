# FEVER — Factual vs Non-Factual (easy beginner code)

**Main file to learn / explain:** [`fever_easy.py`](fever_easy.py)

## Size-specific result READMEs

| File | Contents |
|------|----------|
| [`README_4k.md`](README_4k.md) | All embedders × classifiers × tuning methods for **balanced 4k** |
| [`README_67k.md`](README_67k.md) | All embedders × classifiers × tuning methods for **balanced 67k** |

Regenerate after new runs: `python generate_size_readmes.py`

It is one script with clear steps:

1. Load **balanced** data (equal factual / non-factual)
2. Turn claims into numbers (**TF-IDF@500** or **MiniLM / BERT / RoBERTa / T5**)
3. Train classifiers (Default + Grid / Random / Bayesian)
4. List claims **every** classifier got wrong
5. Save Excel

## Quick run (one size + one embedder)

```bash
source .venv/bin/activate

# 2,000 factual + 2,000 non-factual
python fever_easy.py --size 4k  --embedder tfidf
python fever_easy.py --size 4k  --embedder minilm
python fever_easy.py --size 4k  --embedder bert
python fever_easy.py --size 4k  --embedder roberta
python fever_easy.py --size 4k  --embedder t5

# full balanced (~67,134)
python fever_easy.py --size 67k --embedder tfidf
python fever_easy.py --size 67k --embedder minilm
# ... bert / roberta / t5
```

Or use the thin wrappers:

```bash
python balanced_4k/run_tfidf.py
python balanced_67134/run_minilm.py
```

Run everything:

```bash
# faster overnight (Default only on 67k; full tuning on 4k):
bash run_all_easy.sh

# Default-only for both sizes (quickest complete matrix):
MODES_4K=Default MODES_67K=Default bash run_all_easy.sh
```

## Master Excel (all sizes × embedders)

After any run, this file is refreshed:

`results_easy/MASTER_all_sizes_embedders.xlsx`

| Sheet | Meaning |
|---|---|
| `Summary` | overview table |
| `BestAcc by Size x Embedder` | comparison pivot |
| `4k_tfidf`, `4k_minilm`, … | all classifier results |
| `67k_tfidf`, … | same for full balanced |
| `AM_4k_tfidf`, … | always-misclassified claims |

Per-run files also live under:

`results_easy/<4k|67k>/<embedder>/results_*.xlsx`

## Faculty checklist

| Ask | Covered by |
|---|---|
| TF-IDF dim 500 | `--embedder tfidf` |
| MiniLM, T5, BERT, RoBERTa | matching `--embedder` |
| Always-misclassified phrases | sheet `Always Misclassified` / `AM_*` |
| Balanced >50k | `--size 67k` |

## Cross-fold data-exchange stacking (separate experiment)

**Show faculty:** [`cross_fold_easy.py`](cross_fold_easy.py) + diagrams + **results vs normal stacking** in [`CROSS_FOLD_STACKING_README.md`](CROSS_FOLD_STACKING_README.md) (§7).

```bash
python cross_fold_easy.py --size 4k --embedder tfidf --all-ratios
python -m pytest tests/ -q
```

| Results | Path |
|---|---|
| Cross-fold (all sizes × embedders) | `results_cross_fold/MASTER_cross_fold_stacking.xlsx` |
| Normal stacking (`fever_easy`) | `results_easy/MASTER_all_sizes_embedders.xlsx` |
| **17-base normal vs cross-fold (final)** | [`STACKING_17_FINAL_COMPARISON.md`](STACKING_17_FINAL_COMPARISON.md) + `results_stacking_17/MASTER_stacking_17.xlsx` |

## Full tuning outputs (Default / Grid / Random / Bayesian Optuna)

Primary workbooks (updated as the run progresses):

| File | Contents |
|------|----------|
| `results_full/FINAL_all_results.xlsx` | All balanced sizes × embedders × classifiers × 4 tunings |
| `results_full/FINAL_misclassified_ge5.xlsx` | Claims misclassified by **≥5** classifiers (true vs predicted) |
| `results_full/<4k\|67k>/<embedder>/results_*.xlsx` | Per-run detail sheets |

Resume / continue the overnight job (tmux session `fever_full`):

```bash
tmux attach -t fever_full   # or: .venv/bin/python -u run_resilient_driver.py
```
