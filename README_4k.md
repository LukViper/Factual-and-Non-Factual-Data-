# FEVER Results — Balanced **4k**

_Auto-generated from real experiment CSVs on 2026-11-13 08:27. Values are measured, not synthetic._

## Setup

| Property | Value |
| --- | --- |
| Balanced size | **4k** — 2,000 factual + 2,000 non-factual |
| Rows used | ~4,000 |
| Labels | `factual` (0) / `non-factual` (1) |
| Split | 80% train / 20% test, stratified, `random_state=42` |
| Embedders | TF-IDF+SVD-500, MiniLM, BERT, RoBERTa, T5 |
| Classifiers | 17 (LR, Perceptron, LDA, NB, DT, SVM, KNN, AdaBoost, GB, XGB, LGBM, CatBoost, ET, RF, MLP, Voting, Stacking) |
| Tuning methods | Default, Grid Search, Random Search, Bayesian Optimization (Optuna TPE) |
| Primary Excel | `results_full/FINAL_all_results.xlsx` |
| ≥5 misclassified Excel | `results_full/FINAL_misclassified_ge5.xlsx` |

## Coverage 

| Embedder | Classifiers | Tunings present | Rows | Status |
| --- | --- | --- | --- | --- |
| `tfidf` | 17 | Default, Grid Search, Random Search, Bayesian Optimization | 68 | complete (4 tunings) |
| `minilm` | 17 | Default, Grid Search, Random Search, Bayesian Optimization | 68 | complete (4 tunings) |
| `bert` | 17 | Default, Grid Search, Random Search, Bayesian Optimization | 50 | partial (50/68 expected) |
| `roberta` | 17 | Default | 17 | Default only (Grid/Random/Bayesian pending) |
| `t5` | 17 | Default | 17 | Default only (Grid/Random/Bayesian pending) |


## Best overall (top 15 by Test Accuracy)

| Embedder | Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t5 | LDA | Default | 0.6400 | 0.6400 | 0.6400 | 0.6400 | GOOD FIT |
| bert | LDA | Grid Search | 0.6400 | 0.6400 | 0.6400 | 0.6400 | GOOD FIT (moderate gap) |
| bert | Logistic Regression | Grid Search | 0.6375 | 0.6375 | 0.6375 | 0.6375 | GOOD FIT |
| bert | LDA | Bayesian Optimization | 0.6362 | 0.6362 | 0.6363 | 0.6362 | GOOD FIT |
| bert | LDA | Random Search | 0.6362 | 0.6362 | 0.6363 | 0.6362 | GOOD FIT |
| bert | Logistic Regression | Random Search | 0.6350 | 0.6350 | 0.6350 | 0.6350 | GOOD FIT |
| bert | Logistic Regression | Bayesian Optimization | 0.6350 | 0.6350 | 0.6350 | 0.6350 | GOOD FIT |
| bert | LightGBM | Grid Search | 0.6312 | 0.6312 | 0.6313 | 0.6313 | OVERFITTING |
| t5 | XGBoost | Default | 0.6312 | 0.6312 | 0.6313 | 0.6312 | OVERFITTING |
| bert | LightGBM | Bayesian Optimization | 0.6288 | 0.6287 | 0.6288 | 0.6287 | OVERFITTING |
| bert | Stacking | Default | 0.6238 | 0.6237 | 0.6239 | 0.6238 | OVERFITTING |
| t5 | LightGBM | Default | 0.6238 | 0.6237 | 0.6238 | 0.6238 | OVERFITTING |
| bert | SVM | Bayesian Optimization | 0.6238 | 0.6237 | 0.6238 | 0.6238 | OVERFITTING |
| t5 | Stacking | Default | 0.6238 | 0.6237 | 0.6239 | 0.6238 | OVERFITTING |
| bert | SVM | Random Search | 0.6238 | 0.6237 | 0.6238 | 0.6238 | OVERFITTING |


## Best configuration per embedder

| Embedder | Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Fit Status |
| --- | --- | --- | --- | --- | --- |
| tfidf | LightGBM | Random Search | 0.5750 | 0.5750 | OVERFITTING |
| minilm | Neural Network (MLP) | Default | 0.6075 | 0.6063 | GOOD FIT (moderate gap) |
| bert | LDA | Grid Search | 0.6400 | 0.6400 | GOOD FIT (moderate gap) |
| roberta | LDA | Default | 0.6175 | 0.6174 | GOOD FIT |
| t5 | LDA | Default | 0.6400 | 0.6400 | GOOD FIT |


## Best Test Accuracy by Embedder × Tuning

| Embedder | Default | Grid Search | Random Search | Bayesian Optimization |
| --- | --- | --- | --- | --- |
| tfidf | 0.5687 | 0.5675 | 0.575 | 0.57 |
| minilm | 0.6075 | 0.5925 | 0.5925 | 0.595 |
| bert | 0.6238 | 0.64 | 0.6362 | 0.6362 |
| roberta | 0.6175 | — | — | — |
| t5 | 0.64 | — | — | — |


## Embedder: `tfidf`

Feature set: **TF-IDF + SVD-500**  

Result rows: **68** | Classifiers: **17** | Tunings: **Default, Grid Search, Random Search, Bayesian Optimization**

Best here: **LightGBM** / **Random Search** — Test Acc `0.5750`, F1 `0.5750`

### Test Accuracy by Classifier × Tuning

| Classifier Name | Default | Grid Search | Random Search | Bayesian Optimization |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.5587 | 0.5387 | 0.5413 | 0.5525 |
| Perceptron | 0.5325 | 0.525 | 0.5312 | 0.5125 |
| LDA | 0.5587 | 0.5363 | 0.5375 | 0.5413 |
| Naive Bayes | 0.5125 | 0.5125 | 0.5075 | 0.515 |
| Decision Tree | 0.5062 | 0.525 | 0.5363 | 0.5363 |
| SVM | 0.5613 | 0.535 | 0.54 | 0.5475 |
| KNN | 0.5125 | 0.5312 | 0.5425 | 0.5288 |
| AdaBoost | 0.5312 | 0.54 | 0.5413 | 0.5587 |
| Gradient Boosting | 0.565 | 0.565 | 0.5537 | 0.5463 |
| XGBoost | 0.5425 | 0.545 | 0.5437 | 0.5537 |
| LightGBM | 0.5337 | 0.5375 | 0.575 | 0.565 |
| CatBoost | 0.5687 | 0.5675 | 0.5663 | 0.57 |
| Extra Trees | 0.525 | 0.5587 | 0.5725 | 0.56 |
| Random Forest | 0.5625 | 0.5437 | 0.57 | 0.5563 |
| Neural Network (MLP) | 0.5188 | 0.5375 | 0.565 | 0.5663 |
| Voting | 0.5525 | 0.5275 | 0.5575 | 0.5425 |
| Stacking | 0.5625 | 0.5613 | 0.55 | 0.5537 |

### Full metrics (all classifier × tuning rows)

| Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Training Accuracy | Train-Test Gap | Stratified K-Fold Mean | Overfitting Risk | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Default | 0.5587 | 0.5582 | 0.5590 | 0.5588 | 0.7031 | 0.1444 | 0.5466 | MODERATE | GOOD FIT (moderate gap) |
| Logistic Regression | Grid Search | 0.5387 | 0.5381 | 0.5390 | 0.5388 | 0.6447 | 0.1059 | 0.5497 | MODERATE | GOOD FIT (moderate gap) |
| Logistic Regression | Random Search | 0.5413 | 0.5407 | 0.5414 | 0.5413 | 0.6619 | 0.1206 | 0.5594 | MODERATE | GOOD FIT (moderate gap) |
| Logistic Regression | Bayesian Optimization | 0.5525 | 0.5524 | 0.5525 | 0.5525 | 0.6209 | 0.0684 | 0.5384 | LOW | GOOD FIT |
| Perceptron | Default | 0.5325 | 0.5325 | 0.5325 | 0.5325 | 0.6106 | 0.0781 | 0.5363 | MODERATE | GOOD FIT (moderate gap) |
| Perceptron | Grid Search | 0.5250 | 0.5250 | 0.5250 | 0.5250 | 0.6041 | 0.0791 | 0.5359 | MODERATE | GOOD FIT (moderate gap) |
| Perceptron | Random Search | 0.5312 | 0.5295 | 0.5317 | 0.5312 | 0.5559 | 0.0247 | 0.5244 | LOW | GOOD FIT |
| Perceptron | Bayesian Optimization | 0.5125 | 0.5125 | 0.5125 | 0.5125 | 0.6072 | 0.0947 | 0.5394 | MODERATE | GOOD FIT (moderate gap) |
| LDA | Default | 0.5587 | 0.5582 | 0.5590 | 0.5588 | 0.6984 | 0.1397 | 0.5437 | MODERATE | GOOD FIT (moderate gap) |
| LDA | Grid Search | 0.5363 | 0.5356 | 0.5364 | 0.5363 | 0.6447 | 0.1084 | 0.5478 | MODERATE | GOOD FIT (moderate gap) |
| LDA | Random Search | 0.5375 | 0.5368 | 0.5377 | 0.5375 | 0.6625 | 0.1250 | 0.5572 | MODERATE | GOOD FIT (moderate gap) |
| LDA | Bayesian Optimization | 0.5413 | 0.5402 | 0.5416 | 0.5413 | 0.6634 | 0.1222 | 0.5544 | MODERATE | GOOD FIT (moderate gap) |
| Naive Bayes | Default | 0.5125 | 0.4578 | 0.5209 | 0.5125 | 0.5878 | 0.0753 | 0.5369 | LOW | GOOD FIT |
| Naive Bayes | Grid Search | 0.5125 | 0.4578 | 0.5209 | 0.5125 | 0.5878 | 0.0753 | 0.5369 | LOW | GOOD FIT |
| Naive Bayes | Random Search | 0.5075 | 0.4551 | 0.5122 | 0.5075 | 0.5897 | 0.0822 | 0.5375 | LOW | GOOD FIT |
| Naive Bayes | Bayesian Optimization | 0.5150 | 0.4625 | 0.5246 | 0.5150 | 0.5872 | 0.0722 | 0.5381 | LOW | GOOD FIT |
| Decision Tree | Default | 0.5062 | 0.5062 | 0.5063 | 0.5062 | 0.9988 | 0.4925 | 0.5325 | HIGH | OVERFITTING |
| Decision Tree | Grid Search | 0.5250 | 0.5192 | 0.5263 | 0.5250 | 0.6012 | 0.0762 | 0.5434 | LOW | GOOD FIT |
| Decision Tree | Random Search | 0.5363 | 0.5182 | 0.5427 | 0.5363 | 0.5528 | 0.0166 | 0.5206 | LOW | GOOD FIT |
| Decision Tree | Bayesian Optimization | 0.5363 | 0.5182 | 0.5427 | 0.5363 | 0.5528 | 0.0166 | 0.5319 | LOW | GOOD FIT |
| SVM | Default | 0.5613 | 0.5609 | 0.5614 | 0.5613 | 0.7013 | 0.1400 | 0.5472 | MODERATE | GOOD FIT (moderate gap) |
| SVM | Grid Search | 0.5350 | 0.5343 | 0.5352 | 0.5350 | 0.6444 | 0.1094 | 0.5469 | MODERATE | GOOD FIT (moderate gap) |
| SVM | Random Search | 0.5400 | 0.5396 | 0.5401 | 0.5400 | 0.6619 | 0.1219 | 0.5578 | MODERATE | GOOD FIT (moderate gap) |
| SVM | Bayesian Optimization | 0.5475 | 0.5469 | 0.5477 | 0.5475 | 0.6619 | 0.1144 | 0.5556 | MODERATE | GOOD FIT (moderate gap) |
| KNN | Default | 0.5125 | 0.4827 | 0.5162 | 0.5125 | 0.7150 | 0.2025 | 0.5244 | HIGH | OVERFITTING |
| KNN | Grid Search | 0.5312 | 0.5277 | 0.5322 | 0.5312 | 0.6272 | 0.0959 | 0.5522 | LOW | GOOD FIT |
| KNN | Random Search | 0.5425 | 0.5412 | 0.5430 | 0.5425 | 0.6362 | 0.0938 | 0.5484 | LOW | GOOD FIT |
| KNN | Bayesian Optimization | 0.5288 | 0.5220 | 0.5305 | 0.5287 | 0.6119 | 0.0831 | 0.5447 | LOW | GOOD FIT |
| AdaBoost | Default | 0.5312 | 0.5224 | 0.5338 | 0.5312 | 0.6088 | 0.0775 | 0.5363 | LOW | GOOD FIT |
| AdaBoost | Grid Search | 0.5400 | 0.5400 | 0.5400 | 0.5400 | 0.6188 | 0.0787 | 0.5441 | LOW | GOOD FIT |
| AdaBoost | Random Search | 0.5413 | 0.5341 | 0.5439 | 0.5413 | 0.5619 | 0.0206 | 0.5384 | LOW | GOOD FIT |
| AdaBoost | Bayesian Optimization | 0.5587 | 0.5415 | 0.5691 | 0.5587 | 0.5603 | 0.0016 | 0.5363 | LOW | GOOD FIT |
| Gradient Boosting | Default | 0.5650 | 0.5604 | 0.5679 | 0.5650 | 0.8941 | 0.3291 | 0.5366 | HIGH | OVERFITTING |
| Gradient Boosting | Grid Search | 0.5650 | 0.5647 | 0.5652 | 0.5650 | 0.7522 | 0.1872 | 0.5322 | HIGH | OVERFITTING |
| Gradient Boosting | Random Search | 0.5537 | 0.5537 | 0.5538 | 0.5537 | 0.6122 | 0.0584 | 0.5419 | LOW | GOOD FIT |
| Gradient Boosting | Bayesian Optimization | 0.5463 | 0.5416 | 0.5482 | 0.5463 | 0.6297 | 0.0834 | 0.5419 | MODERATE | GOOD FIT (moderate gap) |
| XGBoost | Default | 0.5425 | 0.5411 | 0.5430 | 0.5425 | 0.9988 | 0.4563 | 0.5347 | HIGH | OVERFITTING |
| XGBoost | Grid Search | 0.5450 | 0.5445 | 0.5452 | 0.5450 | 0.8572 | 0.3122 | 0.5475 | HIGH | OVERFITTING |
| XGBoost | Random Search | 0.5437 | 0.5414 | 0.5447 | 0.5437 | 0.8600 | 0.3163 | 0.5472 | HIGH | OVERFITTING |
| XGBoost | Bayesian Optimization | 0.5537 | 0.5468 | 0.5573 | 0.5537 | 0.6775 | 0.1238 | 0.5466 | MODERATE | GOOD FIT (moderate gap) |
| LightGBM | Default | 0.5337 | 0.5317 | 0.5344 | 0.5337 | 0.9988 | 0.4650 | 0.5525 | HIGH | OVERFITTING |
| LightGBM | Grid Search | 0.5375 | 0.5355 | 0.5381 | 0.5375 | 0.9988 | 0.4613 | 0.5456 | HIGH | OVERFITTING |
| LightGBM | Random Search | 0.5750 | 0.5750 | 0.5750 | 0.5750 | 0.9988 | 0.4238 | 0.5453 | HIGH | OVERFITTING |
| LightGBM | Bayesian Optimization | 0.5650 | 0.5599 | 0.5682 | 0.5650 | 0.8638 | 0.2988 | 0.5388 | HIGH | OVERFITTING |
| CatBoost | Default | 0.5687 | 0.5655 | 0.5709 | 0.5687 | 0.9925 | 0.4238 | 0.5397 | HIGH | OVERFITTING |
| CatBoost | Grid Search | 0.5675 | 0.5675 | 0.5675 | 0.5675 | 0.8166 | 0.2491 | 0.5591 | HIGH | OVERFITTING |
| CatBoost | Random Search | 0.5663 | 0.5660 | 0.5664 | 0.5663 | 0.6512 | 0.0850 | 0.5481 | MODERATE | GOOD FIT (moderate gap) |
| CatBoost | Bayesian Optimization | 0.5700 | 0.5695 | 0.5703 | 0.5700 | 0.6766 | 0.1066 | 0.5475 | MODERATE | GOOD FIT (moderate gap) |
| Extra Trees | Default | 0.5250 | 0.5204 | 0.5260 | 0.5250 | 0.9988 | 0.4738 | 0.5363 | HIGH | OVERFITTING |
| Extra Trees | Grid Search | 0.5587 | 0.5587 | 0.5588 | 0.5587 | 0.9328 | 0.3741 | 0.5463 | HIGH | OVERFITTING |
| Extra Trees | Random Search | 0.5725 | 0.5682 | 0.5755 | 0.5725 | 0.7784 | 0.2059 | 0.5569 | HIGH | OVERFITTING |
| Extra Trees | Bayesian Optimization | 0.5600 | 0.5580 | 0.5611 | 0.5600 | 0.8825 | 0.3225 | 0.5550 | HIGH | OVERFITTING |
| Random Forest | Default | 0.5625 | 0.5585 | 0.5648 | 0.5625 | 0.9988 | 0.4363 | 0.5381 | HIGH | OVERFITTING |
| Random Forest | Grid Search | 0.5437 | 0.5390 | 0.5456 | 0.5437 | 0.9966 | 0.4528 | 0.5534 | HIGH | OVERFITTING |
| Random Forest | Random Search | 0.5700 | 0.5697 | 0.5702 | 0.5700 | 0.8678 | 0.2978 | 0.5506 | HIGH | OVERFITTING |
| Random Forest | Bayesian Optimization | 0.5563 | 0.5547 | 0.5570 | 0.5563 | 0.9525 | 0.3962 | 0.5491 | HIGH | OVERFITTING |
| Neural Network (MLP) | Default | 0.5188 | 0.5180 | 0.5189 | 0.5188 | 0.8166 | 0.2978 | 0.5425 | HIGH | OVERFITTING |
| Neural Network (MLP) | Grid Search | 0.5375 | 0.5373 | 0.5376 | 0.5375 | 0.9381 | 0.4006 | 0.5466 | HIGH | OVERFITTING |
| Neural Network (MLP) | Random Search | 0.5650 | 0.5645 | 0.5653 | 0.5650 | 0.6219 | 0.0569 | 0.5406 | MODERATE | GOOD FIT (moderate gap) |
| Neural Network (MLP) | Bayesian Optimization | 0.5663 | 0.5660 | 0.5664 | 0.5663 | 0.7678 | 0.2016 | 0.5191 | HIGH | OVERFITTING |
| Voting | Default | 0.5525 | 0.5517 | 0.5529 | 0.5525 | 0.7081 | 0.1556 | 0.5475 | MODERATE | GOOD FIT (moderate gap) |
| Voting | Grid Search | 0.5275 | 0.5264 | 0.5278 | 0.5275 | 0.6481 | 0.1206 | 0.5481 | MODERATE | GOOD FIT (moderate gap) |
| Voting | Random Search | 0.5575 | 0.5571 | 0.5577 | 0.5575 | 0.6291 | 0.0716 | 0.5431 | LOW | GOOD FIT |
| Voting | Bayesian Optimization | 0.5425 | 0.5416 | 0.5428 | 0.5425 | 0.6669 | 0.1244 | 0.5528 | MODERATE | GOOD FIT (moderate gap) |
| Stacking | Default | 0.5625 | 0.5617 | 0.5630 | 0.5625 | 0.9219 | 0.3594 | 0.5456 | HIGH | OVERFITTING |
| Stacking | Grid Search | 0.5613 | 0.5607 | 0.5615 | 0.5613 | 0.8394 | 0.2781 | 0.5462 | HIGH | OVERFITTING |
| Stacking | Random Search | 0.5500 | 0.5499 | 0.5500 | 0.5500 | 0.6634 | 0.1134 | 0.5506 | MODERATE | GOOD FIT (moderate gap) |
| Stacking | Bayesian Optimization | 0.5537 | 0.5537 | 0.5538 | 0.5537 | 0.6697 | 0.1159 | 0.5406 | MODERATE | GOOD FIT (moderate gap) |


## Embedder: `minilm`

Feature set: **all-MiniLM-L6-v2**  

Result rows: **68** | Classifiers: **17** | Tunings: **Default, Grid Search, Random Search, Bayesian Optimization**

Best here: **Neural Network (MLP)** / **Default** — Test Acc `0.6075`, F1 `0.6063`

### Test Accuracy by Classifier × Tuning

| Classifier Name | Default | Grid Search | Random Search | Bayesian Optimization |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.5837 | 0.585 | 0.5837 | 0.5837 |
| Perceptron | 0.5637 | 0.5138 | 0.5175 | 0.5787 |
| LDA | 0.5813 | 0.5875 | 0.59 | 0.59 |
| Naive Bayes | 0.5975 | 0.5863 | 0.5725 | 0.5725 |
| Decision Tree | 0.51 | 0.5375 | 0.5337 | 0.54 |
| SVM | 0.5813 | 0.5813 | 0.5837 | 0.5837 |
| KNN | 0.53 | 0.5663 | 0.5775 | 0.5663 |
| AdaBoost | 0.5763 | 0.5875 | 0.59 | 0.5913 |
| Gradient Boosting | 0.5613 | 0.58 | 0.5887 | 0.5825 |
| XGBoost | 0.565 | 0.5913 | 0.59 | 0.595 |
| LightGBM | 0.5763 | 0.5737 | 0.59 | 0.5863 |
| CatBoost | 0.5887 | 0.59 | 0.58 | 0.5875 |
| Extra Trees | 0.5687 | 0.5813 | 0.565 | 0.5675 |
| Random Forest | 0.5775 | 0.57 | 0.5925 | 0.5875 |
| Neural Network (MLP) | 0.6075 | 0.5687 | 0.5425 | 0.5687 |
| Voting | 0.5813 | 0.5813 | 0.58 | 0.5813 |
| Stacking | 0.5887 | 0.5925 | 0.5875 | 0.5787 |

### Full metrics (all classifier × tuning rows)

| Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Training Accuracy | Train-Test Gap | Stratified K-Fold Mean | Overfitting Risk | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Default | 0.5837 | 0.5837 | 0.5838 | 0.5837 | 0.6953 | 0.1116 | 0.5734 | MODERATE | GOOD FIT (moderate gap) |
| Logistic Regression | Grid Search | 0.5850 | 0.5849 | 0.5851 | 0.5850 | 0.6787 | 0.0938 | 0.5741 | MODERATE | GOOD FIT (moderate gap) |
| Logistic Regression | Random Search | 0.5837 | 0.5837 | 0.5838 | 0.5837 | 0.6950 | 0.1112 | 0.5741 | MODERATE | GOOD FIT (moderate gap) |
| Logistic Regression | Bayesian Optimization | 0.5837 | 0.5837 | 0.5838 | 0.5837 | 0.6956 | 0.1119 | 0.5741 | MODERATE | GOOD FIT (moderate gap) |
| Perceptron | Default | 0.5637 | 0.5633 | 0.5640 | 0.5637 | 0.6056 | 0.0419 | 0.5506 | LOW | GOOD FIT |
| Perceptron | Grid Search | 0.5138 | 0.5136 | 0.5138 | 0.5137 | 0.5822 | 0.0684 | 0.5413 | LOW | GOOD FIT |
| Perceptron | Random Search | 0.5175 | 0.5170 | 0.5176 | 0.5175 | 0.6056 | 0.0881 | 0.5522 | LOW | GOOD FIT |
| Perceptron | Bayesian Optimization | 0.5787 | 0.5787 | 0.5788 | 0.5787 | 0.6034 | 0.0247 | 0.5528 | LOW | GOOD FIT |
| LDA | Default | 0.5813 | 0.5812 | 0.5813 | 0.5813 | 0.6950 | 0.1137 | 0.5778 | MODERATE | GOOD FIT (moderate gap) |
| LDA | Grid Search | 0.5875 | 0.5873 | 0.5876 | 0.5875 | 0.6772 | 0.0897 | 0.5794 | MODERATE | GOOD FIT (moderate gap) |
| LDA | Random Search | 0.5900 | 0.5899 | 0.5901 | 0.5900 | 0.6741 | 0.0841 | 0.5737 | MODERATE | GOOD FIT (moderate gap) |
| LDA | Bayesian Optimization | 0.5900 | 0.5899 | 0.5901 | 0.5900 | 0.6741 | 0.0841 | 0.5737 | MODERATE | GOOD FIT (moderate gap) |
| Naive Bayes | Default | 0.5975 | 0.5973 | 0.5977 | 0.5975 | 0.6375 | 0.0400 | 0.5547 | LOW | GOOD FIT |
| Naive Bayes | Grid Search | 0.5863 | 0.5849 | 0.5874 | 0.5862 | 0.6134 | 0.0272 | 0.5553 | LOW | GOOD FIT |
| Naive Bayes | Random Search | 0.5725 | 0.5704 | 0.5739 | 0.5725 | 0.6047 | 0.0322 | 0.5566 | LOW | GOOD FIT |
| Naive Bayes | Bayesian Optimization | 0.5725 | 0.5704 | 0.5739 | 0.5725 | 0.6047 | 0.0322 | 0.5566 | LOW | GOOD FIT |
| Decision Tree | Default | 0.5100 | 0.5096 | 0.5100 | 0.5100 | 1.0000 | 0.4900 | 0.5188 | HIGH | OVERFITTING |
| Decision Tree | Grid Search | 0.5375 | 0.5316 | 0.5395 | 0.5375 | 0.6650 | 0.1275 | 0.5069 | MODERATE | GOOD FIT (moderate gap) |
| Decision Tree | Random Search | 0.5337 | 0.5184 | 0.5387 | 0.5337 | 0.5575 | 0.0238 | 0.5172 | LOW | GOOD FIT |
| Decision Tree | Bayesian Optimization | 0.5400 | 0.5388 | 0.5404 | 0.5400 | 0.6684 | 0.1284 | 0.5088 | MODERATE | GOOD FIT (moderate gap) |
| SVM | Default | 0.5813 | 0.5812 | 0.5813 | 0.5813 | 0.6969 | 0.1156 | 0.5766 | MODERATE | GOOD FIT (moderate gap) |
| SVM | Grid Search | 0.5813 | 0.5812 | 0.5813 | 0.5813 | 0.6956 | 0.1144 | 0.5781 | MODERATE | GOOD FIT (moderate gap) |
| SVM | Random Search | 0.5837 | 0.5837 | 0.5838 | 0.5837 | 0.6975 | 0.1138 | 0.5806 | MODERATE | GOOD FIT (moderate gap) |
| SVM | Bayesian Optimization | 0.5837 | 0.5837 | 0.5838 | 0.5837 | 0.6975 | 0.1138 | 0.5806 | MODERATE | GOOD FIT (moderate gap) |
| KNN | Default | 0.5300 | 0.5299 | 0.5300 | 0.5300 | 0.7266 | 0.1966 | 0.5528 | MODERATE | GOOD FIT (moderate gap) |
| KNN | Grid Search | 0.5663 | 0.5657 | 0.5666 | 0.5663 | 0.6197 | 0.0534 | 0.5594 | LOW | GOOD FIT |
| KNN | Random Search | 0.5775 | 0.5726 | 0.5813 | 0.5775 | 0.6131 | 0.0356 | 0.5653 | LOW | GOOD FIT |
| KNN | Bayesian Optimization | 0.5663 | 0.5657 | 0.5666 | 0.5663 | 0.6197 | 0.0534 | 0.5594 | LOW | GOOD FIT |
| AdaBoost | Default | 0.5763 | 0.5758 | 0.5765 | 0.5763 | 0.6381 | 0.0619 | 0.5328 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Grid Search | 0.5875 | 0.5860 | 0.5888 | 0.5875 | 0.6687 | 0.0812 | 0.5378 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Random Search | 0.5900 | 0.5877 | 0.5921 | 0.5900 | 0.6572 | 0.0672 | 0.5356 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Bayesian Optimization | 0.5913 | 0.5885 | 0.5937 | 0.5913 | 0.6553 | 0.0641 | 0.5331 | MODERATE | GOOD FIT (moderate gap) |
| Gradient Boosting | Default | 0.5613 | 0.5606 | 0.5616 | 0.5613 | 0.9056 | 0.3444 | 0.5541 | HIGH | OVERFITTING |
| Gradient Boosting | Grid Search | 0.5800 | 0.5800 | 0.5800 | 0.5800 | 0.7825 | 0.2025 | 0.5466 | HIGH | OVERFITTING |
| Gradient Boosting | Random Search | 0.5887 | 0.5886 | 0.5889 | 0.5887 | 0.6478 | 0.0591 | 0.5456 | MODERATE | GOOD FIT (moderate gap) |
| Gradient Boosting | Bayesian Optimization | 0.5825 | 0.5824 | 0.5826 | 0.5825 | 0.6528 | 0.0703 | 0.5469 | MODERATE | GOOD FIT (moderate gap) |
| XGBoost | Default | 0.5650 | 0.5650 | 0.5650 | 0.5650 | 1.0000 | 0.4350 | 0.5559 | HIGH | OVERFITTING |
| XGBoost | Grid Search | 0.5913 | 0.5912 | 0.5913 | 0.5913 | 0.9150 | 0.3237 | 0.5625 | HIGH | OVERFITTING |
| XGBoost | Random Search | 0.5900 | 0.5898 | 0.5902 | 0.5900 | 0.6666 | 0.0766 | 0.5513 | MODERATE | GOOD FIT (moderate gap) |
| XGBoost | Bayesian Optimization | 0.5950 | 0.5946 | 0.5954 | 0.5950 | 0.9497 | 0.3547 | 0.5622 | HIGH | OVERFITTING |
| LightGBM | Default | 0.5763 | 0.5762 | 0.5763 | 0.5762 | 1.0000 | 0.4237 | 0.5434 | HIGH | OVERFITTING |
| LightGBM | Grid Search | 0.5737 | 0.5736 | 0.5738 | 0.5737 | 1.0000 | 0.4263 | 0.5600 | HIGH | OVERFITTING |
| LightGBM | Random Search | 0.5900 | 0.5894 | 0.5905 | 0.5900 | 1.0000 | 0.4100 | 0.5644 | HIGH | OVERFITTING |
| LightGBM | Bayesian Optimization | 0.5863 | 0.5858 | 0.5866 | 0.5862 | 0.9878 | 0.4016 | 0.5637 | HIGH | OVERFITTING |
| CatBoost | Default | 0.5887 | 0.5886 | 0.5889 | 0.5887 | 0.9966 | 0.4078 | 0.5591 | HIGH | OVERFITTING |
| CatBoost | Grid Search | 0.5900 | 0.5899 | 0.5901 | 0.5900 | 0.8659 | 0.2759 | 0.5531 | HIGH | OVERFITTING |
| CatBoost | Random Search | 0.5800 | 0.5797 | 0.5802 | 0.5800 | 0.6794 | 0.0994 | 0.5397 | MODERATE | GOOD FIT (moderate gap) |
| CatBoost | Bayesian Optimization | 0.5875 | 0.5873 | 0.5877 | 0.5875 | 0.6919 | 0.1044 | 0.5441 | MODERATE | GOOD FIT (moderate gap) |
| Extra Trees | Default | 0.5687 | 0.5686 | 0.5688 | 0.5688 | 1.0000 | 0.4313 | 0.5491 | HIGH | OVERFITTING |
| Extra Trees | Grid Search | 0.5813 | 0.5811 | 0.5814 | 0.5813 | 1.0000 | 0.4187 | 0.5653 | HIGH | OVERFITTING |
| Extra Trees | Random Search | 0.5650 | 0.5644 | 0.5654 | 0.5650 | 0.9100 | 0.3450 | 0.5613 | HIGH | OVERFITTING |
| Extra Trees | Bayesian Optimization | 0.5675 | 0.5670 | 0.5678 | 0.5675 | 0.9269 | 0.3594 | 0.5644 | HIGH | OVERFITTING |
| Random Forest | Default | 0.5775 | 0.5775 | 0.5775 | 0.5775 | 1.0000 | 0.4225 | 0.5522 | HIGH | OVERFITTING |
| Random Forest | Grid Search | 0.5700 | 0.5696 | 0.5703 | 0.5700 | 1.0000 | 0.4300 | 0.5616 | HIGH | OVERFITTING |
| Random Forest | Random Search | 0.5925 | 0.5921 | 0.5928 | 0.5925 | 0.9997 | 0.4072 | 0.5547 | HIGH | OVERFITTING |
| Random Forest | Bayesian Optimization | 0.5875 | 0.5867 | 0.5882 | 0.5875 | 1.0000 | 0.4125 | 0.5619 | HIGH | OVERFITTING |
| Neural Network (MLP) | Default | 0.6075 | 0.6063 | 0.6088 | 0.6075 | 0.7653 | 0.1578 | 0.5641 | MODERATE | GOOD FIT (moderate gap) |
| Neural Network (MLP) | Grid Search | 0.5687 | 0.5687 | 0.5688 | 0.5688 | 0.7506 | 0.1819 | 0.5641 | HIGH | OVERFITTING |
| Neural Network (MLP) | Random Search | 0.5425 | 0.5424 | 0.5425 | 0.5425 | 0.7312 | 0.1887 | 0.5450 | MODERATE | GOOD FIT (moderate gap) |
| Neural Network (MLP) | Bayesian Optimization | 0.5687 | 0.5687 | 0.5688 | 0.5688 | 0.7059 | 0.1372 | 0.5491 | MODERATE | GOOD FIT (moderate gap) |
| Voting | Default | 0.5813 | 0.5812 | 0.5813 | 0.5813 | 0.7006 | 0.1194 | 0.5778 | MODERATE | GOOD FIT (moderate gap) |
| Voting | Grid Search | 0.5813 | 0.5812 | 0.5813 | 0.5813 | 0.7000 | 0.1187 | 0.5787 | MODERATE | GOOD FIT (moderate gap) |
| Voting | Random Search | 0.5800 | 0.5800 | 0.5800 | 0.5800 | 0.7022 | 0.1222 | 0.5794 | MODERATE | GOOD FIT (moderate gap) |
| Voting | Bayesian Optimization | 0.5813 | 0.5812 | 0.5813 | 0.5813 | 0.7056 | 0.1244 | 0.5775 | MODERATE | GOOD FIT (moderate gap) |
| Stacking | Default | 0.5887 | 0.5887 | 0.5888 | 0.5887 | 1.0000 | 0.4113 | 0.5816 | HIGH | OVERFITTING |
| Stacking | Grid Search | 0.5925 | 0.5925 | 0.5925 | 0.5925 | 0.9866 | 0.3941 | 0.5787 | HIGH | OVERFITTING |
| Stacking | Random Search | 0.5875 | 0.5873 | 0.5876 | 0.5875 | 0.7034 | 0.1159 | 0.5847 | MODERATE | GOOD FIT (moderate gap) |
| Stacking | Bayesian Optimization | 0.5787 | 0.5786 | 0.5788 | 0.5787 | 0.7019 | 0.1231 | 0.5769 | MODERATE | GOOD FIT (moderate gap) |


## Embedder: `bert`

Feature set: **BERT (bert-base-uncased)**  

Result rows: **50** | Classifiers: **17** | Tunings: **Default, Grid Search, Random Search, Bayesian Optimization**

Best here: **LDA** / **Grid Search** — Test Acc `0.6400`, F1 `0.6400`

### Test Accuracy by Classifier × Tuning

| Classifier Name | Default | Grid Search | Random Search | Bayesian Optimization |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.6188 | 0.6375 | 0.635 | 0.635 |
| Perceptron | 0.5863 | 0.5775 | 0.5613 | 0.58 |
| LDA | 0.6088 | 0.64 | 0.6362 | 0.6362 |
| Naive Bayes | 0.6038 | 0.5962 | 0.6012 | 0.6012 |
| Decision Tree | 0.5363 | 0.535 | 0.5525 | 0.5725 |
| SVM | 0.6212 | 0.6225 | 0.6238 | 0.6238 |
| KNN | 0.5687 | 0.5938 | 0.585 | 0.5863 |
| AdaBoost | 0.5737 | 0.6025 | 0.6038 | 0.6088 |
| Gradient Boosting | 0.6075 | 0.6125 | 0.6075 | 0.5962 |
| XGBoost | 0.6088 | 0.6175 | 0.61 | 0.6175 |
| LightGBM | 0.6112 | 0.6312 | 0.6125 | 0.6288 |
| CatBoost | 0.6 | — | — | — |
| Extra Trees | 0.6175 | — | — | — |
| Random Forest | 0.62 | — | — | — |
| Neural Network (MLP) | 0.6125 | — | — | — |
| Voting | 0.6025 | — | — | — |
| Stacking | 0.6238 | — | — | — |

### Full metrics (all classifier × tuning rows)

| Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Training Accuracy | Train-Test Gap | Stratified K-Fold Mean | Overfitting Risk | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Default | 0.6188 | 0.6185 | 0.6190 | 0.6187 | 0.7756 | 0.1569 | 0.5706 | HIGH | OVERFITTING |
| Logistic Regression | Grid Search | 0.6375 | 0.6375 | 0.6375 | 0.6375 | 0.6913 | 0.0538 | 0.6078 | LOW | GOOD FIT |
| Logistic Regression | Random Search | 0.6350 | 0.6350 | 0.6350 | 0.6350 | 0.6891 | 0.0541 | 0.6106 | LOW | GOOD FIT |
| Logistic Regression | Bayesian Optimization | 0.6350 | 0.6350 | 0.6350 | 0.6350 | 0.6891 | 0.0541 | 0.6106 | LOW | GOOD FIT |
| Perceptron | Default | 0.5863 | 0.5862 | 0.5863 | 0.5862 | 0.6806 | 0.0944 | 0.5597 | MODERATE | GOOD FIT (moderate gap) |
| Perceptron | Grid Search | 0.5775 | 0.5773 | 0.5776 | 0.5775 | 0.6669 | 0.0894 | 0.5791 | MODERATE | GOOD FIT (moderate gap) |
| Perceptron | Random Search | 0.5613 | 0.5605 | 0.5617 | 0.5613 | 0.6144 | 0.0531 | 0.5366 | MODERATE | GOOD FIT (moderate gap) |
| Perceptron | Bayesian Optimization | 0.5800 | 0.5799 | 0.5801 | 0.5800 | 0.6547 | 0.0747 | 0.5603 | MODERATE | GOOD FIT (moderate gap) |
| LDA | Default | 0.6088 | 0.6087 | 0.6088 | 0.6088 | 0.7666 | 0.1578 | 0.5694 | HIGH | OVERFITTING |
| LDA | Grid Search | 0.6400 | 0.6400 | 0.6400 | 0.6400 | 0.7003 | 0.0603 | 0.6069 | MODERATE | GOOD FIT (moderate gap) |
| LDA | Random Search | 0.6362 | 0.6362 | 0.6363 | 0.6362 | 0.6884 | 0.0522 | 0.6097 | LOW | GOOD FIT |
| LDA | Bayesian Optimization | 0.6362 | 0.6362 | 0.6363 | 0.6362 | 0.6884 | 0.0522 | 0.6097 | LOW | GOOD FIT |
| Naive Bayes | Default | 0.6038 | 0.6035 | 0.6040 | 0.6038 | 0.6297 | 0.0259 | 0.5800 | LOW | GOOD FIT |
| Naive Bayes | Grid Search | 0.5962 | 0.5950 | 0.5975 | 0.5962 | 0.6325 | 0.0363 | 0.5806 | LOW | GOOD FIT |
| Naive Bayes | Random Search | 0.6012 | 0.6010 | 0.6015 | 0.6012 | 0.6288 | 0.0275 | 0.5794 | LOW | GOOD FIT |
| Naive Bayes | Bayesian Optimization | 0.6012 | 0.6010 | 0.6015 | 0.6012 | 0.6288 | 0.0275 | 0.5794 | LOW | GOOD FIT |
| Decision Tree | Default | 0.5363 | 0.5362 | 0.5363 | 0.5363 | 1.0000 | 0.4637 | 0.5191 | HIGH | OVERFITTING |
| Decision Tree | Grid Search | 0.5350 | 0.5325 | 0.5358 | 0.5350 | 0.6444 | 0.1094 | 0.5256 | MODERATE | GOOD FIT (moderate gap) |
| Decision Tree | Random Search | 0.5525 | 0.5516 | 0.5529 | 0.5525 | 0.5878 | 0.0353 | 0.5172 | LOW | GOOD FIT |
| Decision Tree | Bayesian Optimization | 0.5725 | 0.5720 | 0.5729 | 0.5725 | 0.5594 | -0.0131 | 0.5253 | LOW | GOOD FIT |
| SVM | Default | 0.6212 | 0.6211 | 0.6215 | 0.6212 | 0.7716 | 0.1503 | 0.5697 | HIGH | OVERFITTING |
| SVM | Grid Search | 0.6225 | 0.6223 | 0.6227 | 0.6225 | 0.7719 | 0.1494 | 0.5713 | HIGH | OVERFITTING |
| SVM | Random Search | 0.6238 | 0.6237 | 0.6238 | 0.6238 | 0.7709 | 0.1472 | 0.5775 | HIGH | OVERFITTING |
| SVM | Bayesian Optimization | 0.6238 | 0.6237 | 0.6238 | 0.6238 | 0.7709 | 0.1472 | 0.5775 | HIGH | OVERFITTING |
| KNN | Default | 0.5687 | 0.5687 | 0.5688 | 0.5687 | 0.7259 | 0.1572 | 0.5513 | MODERATE | GOOD FIT (moderate gap) |
| KNN | Grid Search | 0.5938 | 0.5930 | 0.5945 | 0.5938 | 0.6200 | 0.0262 | 0.5628 | LOW | GOOD FIT |
| KNN | Random Search | 0.5850 | 0.5844 | 0.5855 | 0.5850 | 0.6269 | 0.0419 | 0.5553 | LOW | GOOD FIT |
| KNN | Bayesian Optimization | 0.5863 | 0.5813 | 0.5905 | 0.5862 | 0.6162 | 0.0300 | 0.5541 | LOW | GOOD FIT |
| AdaBoost | Default | 0.5737 | 0.5737 | 0.5738 | 0.5737 | 0.6641 | 0.0903 | 0.5591 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Grid Search | 0.6025 | 0.6019 | 0.6032 | 0.6025 | 0.6831 | 0.0806 | 0.5703 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Random Search | 0.6038 | 0.6037 | 0.6038 | 0.6038 | 0.6697 | 0.0659 | 0.5659 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Bayesian Optimization | 0.6088 | 0.6087 | 0.6088 | 0.6088 | 0.6797 | 0.0709 | 0.5659 | MODERATE | GOOD FIT (moderate gap) |
| Gradient Boosting | Default | 0.6075 | 0.6074 | 0.6076 | 0.6075 | 0.9072 | 0.2997 | 0.5687 | HIGH | OVERFITTING |
| Gradient Boosting | Grid Search | 0.6125 | 0.6125 | 0.6125 | 0.6125 | 0.7844 | 0.1719 | 0.5703 | HIGH | OVERFITTING |
| Gradient Boosting | Random Search | 0.6075 | 0.6075 | 0.6075 | 0.6075 | 0.6778 | 0.0703 | 0.5797 | MODERATE | GOOD FIT (moderate gap) |
| Gradient Boosting | Bayesian Optimization | 0.5962 | 0.5962 | 0.5963 | 0.5962 | 0.6600 | 0.0638 | 0.5713 | LOW | GOOD FIT |
| XGBoost | Default | 0.6088 | 0.6087 | 0.6088 | 0.6088 | 1.0000 | 0.3912 | 0.5813 | HIGH | OVERFITTING |
| XGBoost | Grid Search | 0.6175 | 0.6175 | 0.6175 | 0.6175 | 0.9203 | 0.3028 | 0.5706 | HIGH | OVERFITTING |
| XGBoost | Random Search | 0.6100 | 0.6098 | 0.6102 | 0.6100 | 0.6691 | 0.0591 | 0.5756 | MODERATE | GOOD FIT (moderate gap) |
| XGBoost | Bayesian Optimization | 0.6175 | 0.6175 | 0.6175 | 0.6175 | 0.9391 | 0.3216 | 0.5772 | HIGH | OVERFITTING |
| LightGBM | Default | 0.6112 | 0.6112 | 0.6113 | 0.6113 | 1.0000 | 0.3888 | 0.5841 | HIGH | OVERFITTING |
| LightGBM | Grid Search | 0.6312 | 0.6312 | 0.6313 | 0.6313 | 1.0000 | 0.3688 | 0.5797 | HIGH | OVERFITTING |
| LightGBM | Random Search | 0.6125 | 0.6125 | 0.6125 | 0.6125 | 1.0000 | 0.3875 | 0.5863 | HIGH | OVERFITTING |
| LightGBM | Bayesian Optimization | 0.6288 | 0.6287 | 0.6288 | 0.6287 | 1.0000 | 0.3712 | 0.5912 | HIGH | OVERFITTING |
| CatBoost | Default | 0.6000 | 0.6000 | 0.6000 | 0.6000 | 0.9944 | 0.3944 | 0.5813 | HIGH | OVERFITTING |
| Extra Trees | Default | 0.6175 | 0.6175 | 0.6175 | 0.6175 | 1.0000 | 0.3825 | 0.5869 | HIGH | OVERFITTING |
| Random Forest | Default | 0.6200 | 0.6200 | 0.6200 | 0.6200 | 1.0000 | 0.3800 | 0.5894 | HIGH | OVERFITTING |
| Neural Network (MLP) | Default | 0.6125 | 0.6125 | 0.6125 | 0.6125 | 0.9628 | 0.3503 | 0.5956 | HIGH | OVERFITTING |
| Voting | Default | 0.6025 | 0.6023 | 0.6027 | 0.6025 | 0.7756 | 0.1731 | 0.5853 | MODERATE | GOOD FIT (moderate gap) |
| Stacking | Default | 0.6238 | 0.6237 | 0.6239 | 0.6238 | 1.0000 | 0.3762 | 0.6000 | HIGH | OVERFITTING |


## Embedder: `roberta`

Feature set: **DistilBERT CLS-pool (RoBERTa slot)**  

Result rows: **17** | Classifiers: **17** | Tunings: **Default**

Best here: **LDA** / **Default** — Test Acc `0.6175`, F1 `0.6174`

### Test Accuracy by Classifier × Tuning

| Classifier Name | Default |
| --- | --- |
| Logistic Regression | 0.57 |
| Perceptron | 0.5687 |
| LDA | 0.6175 |
| Naive Bayes | 0.5775 |
| Decision Tree | 0.5387 |
| SVM | 0.5663 |
| KNN | 0.55 |
| AdaBoost | 0.57 |
| Gradient Boosting | 0.5913 |
| XGBoost | 0.615 |
| LightGBM | 0.5988 |
| CatBoost | 0.605 |
| Extra Trees | 0.6062 |
| Random Forest | 0.5962 |
| Neural Network (MLP) | 0.5988 |
| Voting | 0.5725 |
| Stacking | 0.6012 |

### Full metrics (all classifier × tuning rows)

| Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Training Accuracy | Train-Test Gap | Stratified K-Fold Mean | Overfitting Risk | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Default | 0.5700 | 0.5700 | 0.5700 | 0.5700 | 0.7619 | 0.1919 | 0.5775 | MODERATE | GOOD FIT (moderate gap) |
| Perceptron | Default | 0.5687 | 0.5686 | 0.5689 | 0.5687 | 0.6556 | 0.0869 | 0.5622 | LOW | GOOD FIT |
| LDA | Default | 0.6175 | 0.6174 | 0.6176 | 0.6175 | 0.6822 | 0.0647 | 0.6028 | LOW | GOOD FIT |
| Naive Bayes | Default | 0.5775 | 0.5744 | 0.5798 | 0.5775 | 0.5637 | -0.0138 | 0.5453 | LOW | GOOD FIT |
| Decision Tree | Default | 0.5387 | 0.5382 | 0.5389 | 0.5388 | 1.0000 | 0.4613 | 0.5328 | HIGH | OVERFITTING |
| SVM | Default | 0.5663 | 0.5662 | 0.5663 | 0.5662 | 0.7622 | 0.1959 | 0.5756 | MODERATE | GOOD FIT (moderate gap) |
| KNN | Default | 0.5500 | 0.5491 | 0.5504 | 0.5500 | 0.7244 | 0.1744 | 0.5550 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Default | 0.5700 | 0.5700 | 0.5700 | 0.5700 | 0.6522 | 0.0822 | 0.5600 | LOW | GOOD FIT |
| Gradient Boosting | Default | 0.5913 | 0.5912 | 0.5913 | 0.5913 | 0.8919 | 0.3006 | 0.5747 | HIGH | OVERFITTING |
| XGBoost | Default | 0.6150 | 0.6150 | 0.6150 | 0.6150 | 1.0000 | 0.3850 | 0.5850 | HIGH | OVERFITTING |
| LightGBM | Default | 0.5988 | 0.5987 | 0.5988 | 0.5988 | 1.0000 | 0.4012 | 0.5862 | HIGH | OVERFITTING |
| CatBoost | Default | 0.6050 | 0.6050 | 0.6050 | 0.6050 | 0.9909 | 0.3859 | 0.5816 | HIGH | OVERFITTING |
| Extra Trees | Default | 0.6062 | 0.6061 | 0.6064 | 0.6062 | 1.0000 | 0.3938 | 0.5753 | HIGH | OVERFITTING |
| Random Forest | Default | 0.5962 | 0.5962 | 0.5963 | 0.5962 | 1.0000 | 0.4038 | 0.5969 | HIGH | OVERFITTING |
| Neural Network (MLP) | Default | 0.5988 | 0.5981 | 0.5994 | 0.5988 | 0.9066 | 0.3078 | 0.5991 | HIGH | OVERFITTING |
| Voting | Default | 0.5725 | 0.5723 | 0.5726 | 0.5725 | 0.7703 | 0.1978 | 0.5794 | MODERATE | GOOD FIT (moderate gap) |
| Stacking | Default | 0.6012 | 0.6012 | 0.6013 | 0.6013 | 1.0000 | 0.3988 | 0.5900 | HIGH | OVERFITTING |


## Embedder: `t5`

Feature set: **DistilBERT mean-pool **  

Result rows: **17** | Classifiers: **17** | Tunings: **Default**

Best here: **LDA** / **Default** — Test Acc `0.6400`, F1 `0.6400`

### Test Accuracy by Classifier × Tuning

| Classifier Name | Default |
| --- | --- |
| Logistic Regression | 0.6038 |
| Perceptron | 0.58 |
| LDA | 0.64 |
| Naive Bayes | 0.5988 |
| Decision Tree | 0.4975 |
| SVM | 0.6012 |
| KNN | 0.6 |
| AdaBoost | 0.5863 |
| Gradient Boosting | 0.6162 |
| XGBoost | 0.6312 |
| LightGBM | 0.6238 |
| CatBoost | 0.6 |
| Extra Trees | 0.6175 |
| Random Forest | 0.62 |
| Neural Network (MLP) | 0.6125 |
| Voting | 0.6025 |
| Stacking | 0.6238 |

### Full metrics (all classifier × tuning rows)

| Classifier Name | Tuning | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Training Accuracy | Train-Test Gap | Stratified K-Fold Mean | Overfitting Risk | Fit Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Default | 0.6038 | 0.6037 | 0.6038 | 0.6038 | 0.7666 | 0.1628 | 0.5859 | MODERATE | GOOD FIT (moderate gap) |
| Perceptron | Default | 0.5800 | 0.5800 | 0.5800 | 0.5800 | 0.7013 | 0.1213 | 0.5678 | MODERATE | GOOD FIT (moderate gap) |
| LDA | Default | 0.6400 | 0.6400 | 0.6400 | 0.6400 | 0.6847 | 0.0447 | 0.6034 | LOW | GOOD FIT |
| Naive Bayes | Default | 0.5988 | 0.5974 | 0.6001 | 0.5988 | 0.5816 | -0.0172 | 0.5644 | LOW | GOOD FIT |
| Decision Tree | Default | 0.4975 | 0.4973 | 0.4975 | 0.4975 | 1.0000 | 0.5025 | 0.5234 | HIGH | OVERFITTING |
| SVM | Default | 0.6012 | 0.6012 | 0.6013 | 0.6013 | 0.7650 | 0.1638 | 0.5806 | MODERATE | GOOD FIT (moderate gap) |
| KNN | Default | 0.6000 | 0.5991 | 0.6009 | 0.6000 | 0.7291 | 0.1291 | 0.5678 | MODERATE | GOOD FIT (moderate gap) |
| AdaBoost | Default | 0.5863 | 0.5862 | 0.5863 | 0.5862 | 0.6597 | 0.0734 | 0.5584 | LOW | GOOD FIT |
| Gradient Boosting | Default | 0.6162 | 0.6162 | 0.6163 | 0.6162 | 0.8934 | 0.2772 | 0.5734 | HIGH | OVERFITTING |
| XGBoost | Default | 0.6312 | 0.6312 | 0.6313 | 0.6312 | 1.0000 | 0.3688 | 0.5656 | HIGH | OVERFITTING |
| LightGBM | Default | 0.6238 | 0.6237 | 0.6238 | 0.6238 | 1.0000 | 0.3762 | 0.5781 | HIGH | OVERFITTING |
| CatBoost | Default | 0.6000 | 0.6000 | 0.6000 | 0.6000 | 0.9944 | 0.3944 | 0.5813 | HIGH | OVERFITTING |
| Extra Trees | Default | 0.6175 | 0.6175 | 0.6175 | 0.6175 | 1.0000 | 0.3825 | 0.5869 | HIGH | OVERFITTING |
| Random Forest | Default | 0.6200 | 0.6200 | 0.6200 | 0.6200 | 1.0000 | 0.3800 | 0.5894 | HIGH | OVERFITTING |
| Neural Network (MLP) | Default | 0.6125 | 0.6125 | 0.6125 | 0.6125 | 0.9628 | 0.3503 | 0.5956 | HIGH | OVERFITTING |
| Voting | Default | 0.6025 | 0.6023 | 0.6027 | 0.6025 | 0.7756 | 0.1731 | 0.5853 | MODERATE | GOOD FIT (moderate gap) |
| Stacking | Default | 0.6238 | 0.6237 | 0.6239 | 0.6238 | 1.0000 | 0.3762 | 0.6000 | HIGH | OVERFITTING |


## Misclassified by ≥5 classifiers

Claims on the held-out test set that were wrong for **at least 5** classifiers 
(best tuning per classifier). Sheets include **true_label vs predicted_label**.

| Embedder | Claims misclassified by ≥5 classifiers | Source |
| --- | ---: | --- |
| `tfidf` | 511 | `results_full/4k/tfidf/misclassified_ge5_tfidf.csv` |
| `minilm` | 472 | `results_full/4k/minilm/misclassified_ge5_minilm.csv` |
| `bert` | 353 | `results_full/4k/bert/misclassified_ge5_bert.csv` |
| `roberta` | (always-misclassified: 14; ge5 not yet) | `results_easy` |
| `t5` | (always-misclassified: 24; ge5 not yet) | `results_easy` |


## Artifact paths

| Kind | Path |
| --- | --- |
| Combined results Excel | `results_full/FINAL_all_results.xlsx` |
| ≥5 misclassified Excel | `results_full/FINAL_misclassified_ge5.xlsx` |
| Per-embedder (full tuning) | `results_full/4k/<embedder>/results_<embedder>.xlsx` |
| Per-embedder (easy/Default) | `results_easy/4k/<embedder>/results_4k_<embedder>.xlsx` |
| Master easy workbook | `results_easy/MASTER_all_sizes_embedders.xlsx` |

## How to reproduce

```bash
source .venv/bin/activate
# Full 4 tunings for this size:
python ML.py --size 4k --features tfidf,minilm,bert,roberta,t5 \
  --modes "Default,Grid Search,Random Search,Bayesian Optimization" --resume
python ML.py --build-final
```
