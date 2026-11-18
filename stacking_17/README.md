# Stacking with 17-suite bases + top-performing meta

## Design

| Role | What |
|---|---|
| **Bases** | All **15 individual** classifiers from the project’s 17-classifier suite (`Voting` and nested `Stacking` excluded as bases) |
| **Meta** | **Top-K** (default 3) of those same classifiers, ranked by validation accuracy on development data |
| **Methods** | (1) **Normal stacking** — sklearn `StackingClassifier` (2) **Cross-fold exchange stacking** — F1–F5 allocation + E0–E3 exchange |

Same **80% development / 20% holdout** split and features for both methods.

## Run

```bash
source .venv/bin/activate
cd /home/luk_viper/ML

# one cell
python -m stacking_17 --size 4k --embedder tfidf

# full matrix
python -m stacking_17 --all
```

## Outputs

`results_stacking_17/`

- `<size>_<embedder>/normal_stacking.xlsx`
- `<size>_<embedder>/cross_fold_stacking.xlsx`
- `<size>_<embedder>/combined_<size>_<embedder>.xlsx`
- `MASTER_stacking_17.xlsx`
