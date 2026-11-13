# FEVER Results — Balanced **67k**

_Auto-generated from real experiment CSVs on 2026-11-13 08:27. Values are measured, not synthetic._

## Setup

| Property | Value |
| --- | --- |
| Balanced size | **67k** — full minority-class balance (~33,567 + 33,567) |
| Rows used | ~67,134 |
| Labels | `factual` (0) / `non-factual` (1) |
| Split | 80% train / 20% test, stratified, `random_state=42` |
| Embedders | TF-IDF+SVD-500, MiniLM, BERT, RoBERTa, T5 |
| Classifiers | 17 (LR, Perceptron, LDA, NB, DT, SVM, KNN, AdaBoost, GB, XGB, LGBM, CatBoost, ET, RF, MLP, Voting, Stacking) |
| Tuning methods | Default, Grid Search, Random Search, Bayesian Optimization (Optuna TPE) |
| Primary Excel | `results_full/FINAL_all_results.xlsx` |
| ≥5 misclassified Excel | `results_full/FINAL_misclassified_ge5.xlsx` |

## Coverage (what is finished)

| Embedder | Classifiers | Tunings present | Rows | Status |
| --- | --- | --- | --- | --- |
| `tfidf` | 17 | Default | 17 | Default only (Grid/Random/Bayesian pending) |
| `minilm` | 17 | Default | 17 | Default only (Grid/Random/Bayesian pending) |
| `bert` | 17 | Default | 17 | Default only (Grid/Random/Bayesian pending) |
| `roberta` | 17 | Default | 17 | Default only (Grid/Random/Bayesian pending) |
| `t5` | 17 | Default | 17 | Default only (Grid/Random/Bayesian pending) |


## Best overall (top 15 by Test Accuracy)

| Embedder | Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bert | Stacking | Default | 0.6541 | 0.6540 | 0.6542 | 0.6541 | OVERFITTING |
| t5 | Stacking | Default | 0.6541 | 0.6540 | 0.6542 | 0.6541 | OVERFITTING |
| t5 | Neural Network (MLP) | Default | 0.6503 | 0.6501 | 0.6507 | 0.6503 | GOOD FIT |
| bert | Neural Network (MLP) | Default | 0.6503 | 0.6501 | 0.6507 | 0.6503 | GOOD FIT |
| bert | Logistic Regression | Default | 0.6458 | 0.6457 | 0.6459 | 0.6458 | GOOD FIT |
| t5 | Logistic Regression | Default | 0.6458 | 0.6457 | 0.6459 | 0.6458 | GOOD FIT |
| t5 | Voting | Default | 0.6456 | 0.6455 | 0.6456 | 0.6456 | GOOD FIT |
| bert | Voting | Default | 0.6456 | 0.6455 | 0.6456 | 0.6456 | GOOD FIT |
| bert | SVM | Default | 0.6455 | 0.6454 | 0.6456 | 0.6455 | GOOD FIT |
| t5 | SVM | Default | 0.6455 | 0.6454 | 0.6456 | 0.6455 | GOOD FIT |
| roberta | Stacking | Default | 0.6438 | 0.6436 | 0.6441 | 0.6438 | OVERFITTING |
| roberta | Neural Network (MLP) | Default | 0.6406 | 0.6404 | 0.6410 | 0.6407 | GOOD FIT (moderate gap) |
| roberta | SVM | Default | 0.6369 | 0.6368 | 0.6369 | 0.6369 | GOOD FIT |
| roberta | Logistic Regression | Default | 0.6369 | 0.6369 | 0.6370 | 0.6369 | GOOD FIT |
| roberta | Voting | Default | 0.6369 | 0.6369 | 0.6370 | 0.6369 | GOOD FIT |


## Best configuration per embedder

| Embedder | Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Fit Status |
| --- | --- | --- | --- | --- | --- |
| tfidf | LightGBM | Default | 0.5968 | 0.5954 | GOOD FIT (moderate gap) |
| minilm | Stacking | Default | 0.6269 | 0.6268 | OVERFITTING |
| bert | Stacking | Default | 0.6541 | 0.6540 | OVERFITTING |
| roberta | Stacking | Default | 0.6438 | 0.6436 | OVERFITTING |
| t5 | Stacking | Default | 0.6541 | 0.6540 | OVERFITTING |


## Best Test Accuracy by Embedder × Tuning

| Embedder | Default |
| --- | --- |
| tfidf | 0.5968 |
| minilm | 0.6269 |
| bert | 0.6541 |
| roberta | 0.6438 |
| t5 | 0.6541 |


## Embedder: `tfidf`

Feature set: **TF-IDF + SVD-500**  

Result rows: **17** | Classifiers: **17** | Tunings: **Default**

Best here: **LightGBM** / **Default** — Test Acc `0.5968`, F1 `0.5954`

### Test Accuracy by Classifier × Tuning

| Classifier Name | Default |
| --- | --- |
| Logistic Regression | 0.5832 |
| Perceptron | 0.54 |
| LDA | 0.584 |
| Naive Bayes | 0.5404 |
| Decision Tree | 0.5408 |
| SVM | 0.5826 |
| KNN | 0.5641 |
| AdaBoost | 0.5291 |
| Gradient Boosting | 0.5823 |
| XGBoost | 0.5858 |
| LightGBM | 0.5968 |
| CatBoost | 0.5919 |
| Extra Trees | 0.5943 |
| Random Forest | 0.5934 |
| Neural Network (MLP) | 0.5931 |
| Voting | 0.5836 |
| Stacking | 0.5937 |

### Full metrics (all classifier × tuning rows)

| Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Training Accuracy | Train-Test Gap | Stratified K-Fold Mean | Overfitting Risk | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Default | 0.5832 | 0.5820 | 0.5841 | 0.5832 | 0.6025 | 0.0194 | 0.5832 | LOW | GOOD FIT |
| Perceptron | Default | 0.5400 | 0.5397 | 0.5400 | 0.5400 | 0.5352 | -0.0048 | 0.5400 | LOW | GOOD FIT |
| LDA | Default | 0.5840 | 0.5828 | 0.5850 | 0.5841 | 0.6021 | 0.0181 | 0.5840 | LOW | GOOD FIT |
| Naive Bayes | Default | 0.5404 | 0.5400 | 0.5405 | 0.5404 | 0.5458 | 0.0054 | 0.5404 | LOW | GOOD FIT |
| Decision Tree | Default | 0.5408 | 0.5408 | 0.5408 | 0.5408 | 0.9991 | 0.4583 | 0.5408 | HIGH | OVERFITTING |
| SVM | Default | 0.5826 | 0.5813 | 0.5835 | 0.5826 | 0.6023 | 0.0198 | 0.5826 | LOW | GOOD FIT |
| KNN | Default | 0.5641 | 0.5638 | 0.5643 | 0.5641 | 0.7225 | 0.1584 | 0.5641 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Default | 0.5291 | 0.5156 | 0.5327 | 0.5291 | 0.5363 | 0.0072 | 0.5291 | LOW | GOOD FIT |
| Gradient Boosting | Default | 0.5823 | 0.5779 | 0.5858 | 0.5823 | 0.6211 | 0.0389 | 0.5823 | LOW | GOOD FIT |
| XGBoost | Default | 0.5858 | 0.5854 | 0.5860 | 0.5858 | 0.8693 | 0.2835 | 0.5858 | HIGH | OVERFITTING |
| LightGBM | Default | 0.5968 | 0.5954 | 0.5981 | 0.5968 | 0.7391 | 0.1423 | 0.5968 | MODERATE | GOOD FIT (moderate gap) |
| CatBoost | Default | 0.5919 | 0.5913 | 0.5925 | 0.5919 | 0.7497 | 0.1578 | 0.5919 | MODERATE | GOOD FIT (moderate gap) |
| Extra Trees | Default | 0.5943 | 0.5942 | 0.5943 | 0.5942 | 0.9991 | 0.4048 | 0.5943 | HIGH | OVERFITTING |
| Random Forest | Default | 0.5934 | 0.5934 | 0.5934 | 0.5934 | 0.9991 | 0.4057 | 0.5934 | HIGH | OVERFITTING |
| Neural Network (MLP) | Default | 0.5931 | 0.5926 | 0.5936 | 0.5931 | 0.7387 | 0.1455 | 0.5931 | MODERATE | GOOD FIT (moderate gap) |
| Voting | Default | 0.5836 | 0.5825 | 0.5845 | 0.5836 | 0.6054 | 0.0218 | 0.5836 | LOW | GOOD FIT |
| Stacking | Default | 0.5937 | 0.5933 | 0.5941 | 0.5937 | 0.9922 | 0.3985 | 0.5937 | HIGH | OVERFITTING |


## Embedder: `minilm`

Feature set: **all-MiniLM-L6-v2 (reused cache)**  

Result rows: **17** | Classifiers: **17** | Tunings: **Default**

Best here: **Stacking** / **Default** — Test Acc `0.6269`, F1 `0.6268`

### Test Accuracy by Classifier × Tuning

| Classifier Name | Default |
| --- | --- |
| Logistic Regression | 0.6128 |
| Perceptron | 0.531 |
| LDA | 0.6002 |
| Naive Bayes | 0.5772 |
| Decision Tree | 0.5242 |
| SVM | 0.6138 |
| KNN | 0.5531 |
| AdaBoost | 0.5677 |
| Gradient Boosting | 0.5881 |
| XGBoost | 0.5984 |
| LightGBM | 0.6027 |
| CatBoost | 0.5918 |
| Extra Trees | 0.6086 |
| Random Forest | 0.6059 |
| Neural Network (MLP) | 0.6166 |
| Voting | 0.6129 |
| Stacking | 0.6269 |

### Full metrics (all classifier × tuning rows)

| Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Training Accuracy | Train-Test Gap | Stratified K-Fold Mean | Overfitting Risk | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Default | 0.6128 | 0.6127 | 0.6129 | 0.6128 | 0.6250 | 0.0122 | 0.6128 | LOW | GOOD FIT |
| Perceptron | Default | 0.5310 | 0.5307 | 0.5311 | 0.5310 | 0.5439 | 0.0129 | 0.5310 | LOW | GOOD FIT |
| LDA | Default | 0.6002 | 0.5999 | 0.6005 | 0.6002 | 0.6109 | 0.0107 | 0.6002 | LOW | GOOD FIT |
| Naive Bayes | Default | 0.5772 | 0.5764 | 0.5778 | 0.5772 | 0.5884 | 0.0112 | 0.5772 | LOW | GOOD FIT |
| Decision Tree | Default | 0.5242 | 0.5242 | 0.5242 | 0.5242 | 1.0000 | 0.4758 | 0.5242 | HIGH | OVERFITTING |
| SVM | Default | 0.6138 | 0.6137 | 0.6139 | 0.6138 | 0.6248 | 0.0110 | 0.6138 | LOW | GOOD FIT |
| KNN | Default | 0.5531 | 0.5417 | 0.5590 | 0.5531 | 0.7032 | 0.1500 | 0.5531 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Default | 0.5677 | 0.5676 | 0.5678 | 0.5677 | 0.5852 | 0.0175 | 0.5677 | LOW | GOOD FIT |
| Gradient Boosting | Default | 0.5881 | 0.5875 | 0.5885 | 0.5881 | 0.6426 | 0.0545 | 0.5881 | LOW | GOOD FIT |
| XGBoost | Default | 0.5984 | 0.5984 | 0.5984 | 0.5984 | 0.9077 | 0.3092 | 0.5984 | HIGH | OVERFITTING |
| LightGBM | Default | 0.6027 | 0.6025 | 0.6028 | 0.6027 | 0.7598 | 0.1572 | 0.6027 | MODERATE | GOOD FIT (moderate gap) |
| CatBoost | Default | 0.5918 | 0.5918 | 0.5918 | 0.5918 | 0.7754 | 0.1836 | 0.5918 | MODERATE | GOOD FIT (moderate gap) |
| Extra Trees | Default | 0.6086 | 0.6085 | 0.6087 | 0.6086 | 1.0000 | 0.3914 | 0.6086 | HIGH | OVERFITTING |
| Random Forest | Default | 0.6059 | 0.6058 | 0.6060 | 0.6059 | 1.0000 | 0.3941 | 0.6059 | HIGH | OVERFITTING |
| Neural Network (MLP) | Default | 0.6166 | 0.6166 | 0.6166 | 0.6166 | 0.7643 | 0.1477 | 0.6166 | MODERATE | GOOD FIT (moderate gap) |
| Voting | Default | 0.6129 | 0.6128 | 0.6129 | 0.6129 | 0.6271 | 0.0142 | 0.6129 | LOW | GOOD FIT |
| Stacking | Default | 0.6269 | 0.6268 | 0.6270 | 0.6269 | 0.9997 | 0.3728 | 0.6269 | HIGH | OVERFITTING |


## Embedder: `bert`

Feature set: **DistilBERT mean-pool (BERT-family fallback)**  

Result rows: **17** | Classifiers: **17** | Tunings: **Default**

Best here: **Stacking** / **Default** — Test Acc `0.6541`, F1 `0.6540`

### Test Accuracy by Classifier × Tuning

| Classifier Name | Default |
| --- | --- |
| Logistic Regression | 0.6458 |
| Perceptron | 0.584 |
| LDA | 0.6345 |
| Naive Bayes | 0.5694 |
| Decision Tree | 0.551 |
| SVM | 0.6455 |
| KNN | 0.6181 |
| AdaBoost | 0.602 |
| Gradient Boosting | 0.6231 |
| XGBoost | 0.6247 |
| LightGBM | 0.6354 |
| CatBoost | 0.6363 |
| Extra Trees | 0.633 |
| Random Forest | 0.6305 |
| Neural Network (MLP) | 0.6503 |
| Voting | 0.6456 |
| Stacking | 0.6541 |

### Full metrics (all classifier × tuning rows)

| Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Training Accuracy | Train-Test Gap | Stratified K-Fold Mean | Overfitting Risk | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Default | 0.6458 | 0.6457 | 0.6459 | 0.6458 | 0.6634 | 0.0176 | 0.6458 | LOW | GOOD FIT |
| Perceptron | Default | 0.5840 | 0.5840 | 0.5840 | 0.5840 | 0.5923 | 0.0084 | 0.5840 | LOW | GOOD FIT |
| LDA | Default | 0.6345 | 0.6343 | 0.6347 | 0.6345 | 0.6410 | 0.0065 | 0.6345 | LOW | GOOD FIT |
| Naive Bayes | Default | 0.5694 | 0.5678 | 0.5704 | 0.5694 | 0.5786 | 0.0092 | 0.5694 | LOW | GOOD FIT |
| Decision Tree | Default | 0.5510 | 0.5510 | 0.5510 | 0.5510 | 1.0000 | 0.4490 | 0.5510 | HIGH | OVERFITTING |
| SVM | Default | 0.6455 | 0.6454 | 0.6456 | 0.6455 | 0.6629 | 0.0174 | 0.6455 | LOW | GOOD FIT |
| KNN | Default | 0.6181 | 0.6168 | 0.6197 | 0.6181 | 0.7578 | 0.1398 | 0.6181 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Default | 0.6020 | 0.6015 | 0.6025 | 0.6020 | 0.6210 | 0.0190 | 0.6020 | LOW | GOOD FIT |
| Gradient Boosting | Default | 0.6231 | 0.6226 | 0.6237 | 0.6231 | 0.6604 | 0.0373 | 0.6231 | LOW | GOOD FIT |
| XGBoost | Default | 0.6247 | 0.6247 | 0.6247 | 0.6247 | 0.9155 | 0.2908 | 0.6247 | HIGH | OVERFITTING |
| LightGBM | Default | 0.6354 | 0.6352 | 0.6357 | 0.6354 | 0.7532 | 0.1178 | 0.6354 | MODERATE | GOOD FIT (moderate gap) |
| CatBoost | Default | 0.6363 | 0.6362 | 0.6365 | 0.6363 | 0.7827 | 0.1464 | 0.6363 | MODERATE | GOOD FIT (moderate gap) |
| Extra Trees | Default | 0.6330 | 0.6329 | 0.6330 | 0.6330 | 1.0000 | 0.3670 | 0.6330 | HIGH | OVERFITTING |
| Random Forest | Default | 0.6305 | 0.6305 | 0.6306 | 0.6305 | 1.0000 | 0.3695 | 0.6305 | HIGH | OVERFITTING |
| Neural Network (MLP) | Default | 0.6503 | 0.6501 | 0.6507 | 0.6503 | 0.7279 | 0.0776 | 0.6503 | LOW | GOOD FIT |
| Voting | Default | 0.6456 | 0.6455 | 0.6456 | 0.6456 | 0.6661 | 0.0205 | 0.6456 | LOW | GOOD FIT |
| Stacking | Default | 0.6541 | 0.6540 | 0.6542 | 0.6541 | 0.9990 | 0.3450 | 0.6541 | HIGH | OVERFITTING |


## Embedder: `roberta`

Feature set: **DistilBERT CLS-pool (RoBERTa slot; HF blocked)**  

Result rows: **17** | Classifiers: **17** | Tunings: **Default**

Best here: **Stacking** / **Default** — Test Acc `0.6438`, F1 `0.6436`

### Test Accuracy by Classifier × Tuning

| Classifier Name | Default |
| --- | --- |
| Logistic Regression | 0.6369 |
| Perceptron | 0.5765 |
| LDA | 0.6264 |
| Naive Bayes | 0.5584 |
| Decision Tree | 0.5464 |
| SVM | 0.6369 |
| KNN | 0.6073 |
| AdaBoost | 0.5987 |
| Gradient Boosting | 0.6171 |
| XGBoost | 0.6156 |
| LightGBM | 0.6289 |
| CatBoost | 0.6263 |
| Extra Trees | 0.6264 |
| Random Forest | 0.628 |
| Neural Network (MLP) | 0.6406 |
| Voting | 0.6369 |
| Stacking | 0.6438 |

### Full metrics (all classifier × tuning rows)

| Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Training Accuracy | Train-Test Gap | Stratified K-Fold Mean | Overfitting Risk | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Default | 0.6369 | 0.6369 | 0.6370 | 0.6369 | 0.6518 | 0.0148 | 0.6369 | LOW | GOOD FIT |
| Perceptron | Default | 0.5765 | 0.5764 | 0.5765 | 0.5765 | 0.5769 | 0.0005 | 0.5765 | LOW | GOOD FIT |
| LDA | Default | 0.6264 | 0.6262 | 0.6268 | 0.6264 | 0.6329 | 0.0065 | 0.6264 | LOW | GOOD FIT |
| Naive Bayes | Default | 0.5584 | 0.5562 | 0.5596 | 0.5584 | 0.5605 | 0.0020 | 0.5584 | LOW | GOOD FIT |
| Decision Tree | Default | 0.5464 | 0.5464 | 0.5464 | 0.5464 | 1.0000 | 0.4536 | 0.5464 | HIGH | OVERFITTING |
| SVM | Default | 0.6369 | 0.6368 | 0.6369 | 0.6369 | 0.6518 | 0.0149 | 0.6369 | LOW | GOOD FIT |
| KNN | Default | 0.6073 | 0.6065 | 0.6081 | 0.6073 | 0.7524 | 0.1451 | 0.6073 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Default | 0.5987 | 0.5981 | 0.5993 | 0.5987 | 0.6111 | 0.0123 | 0.5987 | LOW | GOOD FIT |
| Gradient Boosting | Default | 0.6171 | 0.6166 | 0.6178 | 0.6171 | 0.6554 | 0.0383 | 0.6171 | LOW | GOOD FIT |
| XGBoost | Default | 0.6156 | 0.6156 | 0.6156 | 0.6156 | 0.9059 | 0.2903 | 0.6156 | HIGH | OVERFITTING |
| LightGBM | Default | 0.6289 | 0.6285 | 0.6294 | 0.6289 | 0.7512 | 0.1223 | 0.6289 | MODERATE | GOOD FIT (moderate gap) |
| CatBoost | Default | 0.6263 | 0.6262 | 0.6266 | 0.6264 | 0.7765 | 0.1501 | 0.6263 | MODERATE | GOOD FIT (moderate gap) |
| Extra Trees | Default | 0.6264 | 0.6263 | 0.6266 | 0.6264 | 1.0000 | 0.3736 | 0.6264 | HIGH | OVERFITTING |
| Random Forest | Default | 0.6280 | 0.6278 | 0.6282 | 0.6280 | 1.0000 | 0.3720 | 0.6280 | HIGH | OVERFITTING |
| Neural Network (MLP) | Default | 0.6406 | 0.6404 | 0.6410 | 0.6407 | 0.7740 | 0.1333 | 0.6406 | MODERATE | GOOD FIT (moderate gap) |
| Voting | Default | 0.6369 | 0.6369 | 0.6370 | 0.6369 | 0.6555 | 0.0186 | 0.6369 | LOW | GOOD FIT |
| Stacking | Default | 0.6438 | 0.6436 | 0.6441 | 0.6438 | 0.9998 | 0.3560 | 0.6438 | HIGH | OVERFITTING |


## Embedder: `t5`

Feature set: **DistilBERT mean-pool (T5 slot fallback; download sentence-t5-base later)**  

Result rows: **17** | Classifiers: **17** | Tunings: **Default**

Best here: **Stacking** / **Default** — Test Acc `0.6541`, F1 `0.6540`

### Test Accuracy by Classifier × Tuning

| Classifier Name | Default |
| --- | --- |
| Logistic Regression | 0.6458 |
| Perceptron | 0.584 |
| LDA | 0.6345 |
| Naive Bayes | 0.5694 |
| Decision Tree | 0.551 |
| SVM | 0.6455 |
| KNN | 0.6181 |
| AdaBoost | 0.602 |
| Gradient Boosting | 0.6231 |
| XGBoost | 0.6247 |
| LightGBM | 0.6354 |
| CatBoost | 0.6363 |
| Extra Trees | 0.633 |
| Random Forest | 0.6305 |
| Neural Network (MLP) | 0.6503 |
| Voting | 0.6456 |
| Stacking | 0.6541 |

### Full metrics (all classifier × tuning rows)

| Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Training Accuracy | Train-Test Gap | Stratified K-Fold Mean | Overfitting Risk | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Default | 0.6458 | 0.6457 | 0.6459 | 0.6458 | 0.6634 | 0.0176 | 0.6458 | LOW | GOOD FIT |
| Perceptron | Default | 0.5840 | 0.5840 | 0.5840 | 0.5840 | 0.5923 | 0.0084 | 0.5840 | LOW | GOOD FIT |
| LDA | Default | 0.6345 | 0.6343 | 0.6347 | 0.6345 | 0.6410 | 0.0065 | 0.6345 | LOW | GOOD FIT |
| Naive Bayes | Default | 0.5694 | 0.5678 | 0.5704 | 0.5694 | 0.5786 | 0.0092 | 0.5694 | LOW | GOOD FIT |
| Decision Tree | Default | 0.5510 | 0.5510 | 0.5510 | 0.5510 | 1.0000 | 0.4490 | 0.5510 | HIGH | OVERFITTING |
| SVM | Default | 0.6455 | 0.6454 | 0.6456 | 0.6455 | 0.6629 | 0.0174 | 0.6455 | LOW | GOOD FIT |
| KNN | Default | 0.6181 | 0.6168 | 0.6197 | 0.6181 | 0.7578 | 0.1398 | 0.6181 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Default | 0.6020 | 0.6015 | 0.6025 | 0.6020 | 0.6210 | 0.0190 | 0.6020 | LOW | GOOD FIT |
| Gradient Boosting | Default | 0.6231 | 0.6226 | 0.6237 | 0.6231 | 0.6604 | 0.0373 | 0.6231 | LOW | GOOD FIT |
| XGBoost | Default | 0.6247 | 0.6247 | 0.6247 | 0.6247 | 0.9155 | 0.2908 | 0.6247 | HIGH | OVERFITTING |
| LightGBM | Default | 0.6354 | 0.6352 | 0.6357 | 0.6354 | 0.7532 | 0.1178 | 0.6354 | MODERATE | GOOD FIT (moderate gap) |
| CatBoost | Default | 0.6363 | 0.6362 | 0.6365 | 0.6363 | 0.7827 | 0.1464 | 0.6363 | MODERATE | GOOD FIT (moderate gap) |
| Extra Trees | Default | 0.6330 | 0.6329 | 0.6330 | 0.6330 | 1.0000 | 0.3670 | 0.6330 | HIGH | OVERFITTING |
| Random Forest | Default | 0.6305 | 0.6305 | 0.6306 | 0.6305 | 1.0000 | 0.3695 | 0.6305 | HIGH | OVERFITTING |
| Neural Network (MLP) | Default | 0.6503 | 0.6501 | 0.6507 | 0.6503 | 0.7279 | 0.0776 | 0.6503 | LOW | GOOD FIT |
| Voting | Default | 0.6456 | 0.6455 | 0.6456 | 0.6456 | 0.6661 | 0.0205 | 0.6456 | LOW | GOOD FIT |
| Stacking | Default | 0.6541 | 0.6540 | 0.6542 | 0.6541 | 0.9990 | 0.3450 | 0.6541 | HIGH | OVERFITTING |


## Misclassified by ≥5 classifiers

Claims on the held-out test set that were wrong for **at least 5** classifiers 
(best tuning per classifier). Sheets include **true_label vs predicted_label**.

| Embedder | Claims misclassified by ≥5 classifiers | Source |
| --- | ---: | --- |
| `tfidf` | (always-misclassified: 230; ge5 not yet) | `results_easy` |
| `minilm` | (always-misclassified: 262; ge5 not yet) | `results_easy` |
| `bert` | (always-misclassified: 388; ge5 not yet) | `results_easy` |
| `roberta` | (always-misclassified: 362; ge5 not yet) | `results_easy` |
| `t5` | (always-misclassified: 388; ge5 not yet) | `results_easy` |


## Artifact paths

| Kind | Path |
| --- | --- |
| Combined results Excel | `results_full/FINAL_all_results.xlsx` |
| ≥5 misclassified Excel | `results_full/FINAL_misclassified_ge5.xlsx` |
| Per-embedder (full tuning) | `results_full/67k/<embedder>/results_<embedder>.xlsx` |
| Per-embedder (easy/Default) | `results_easy/67k/<embedder>/results_67k_<embedder>.xlsx` |
| Master easy workbook | `results_easy/MASTER_all_sizes_embedders.xlsx` |

## How to reproduce

```bash
source .venv/bin/activate
# Full 4 tunings for this size:
python ML.py --size 67k --features tfidf,minilm,bert,roberta,t5 \
  --modes "Default,Grid Search,Random Search,Bayesian Optimization" --resume
  # add --skip_cv on 67k for memory safety
python ML.py --build-final
```
