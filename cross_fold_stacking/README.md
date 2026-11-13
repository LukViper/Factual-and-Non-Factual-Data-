# Cross-fold data-exchange stacking (package)

**For faculty / viva:** use the one-file demo and diagram README instead:

- [`../cross_fold_easy.py`](../cross_fold_easy.py)
- [`../CROSS_FOLD_STACKING_README.md`](../CROSS_FOLD_STACKING_README.md)

This folder is the full modular implementation used by the demo and by the batch runner.

```bash
python -m cross_fold_stacking --size 4k --embedder tfidf
python -m pytest tests/ -q
```
