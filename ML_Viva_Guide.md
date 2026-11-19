# ML Viva Survival Guide
**A Beginner-Friendly Explanation of All 14 PPTX Files**  
*Course: Machine Learning (22AIE213) | Instructor: Dr. Debanjali Bhattacharya*  
*Prepared for: Viva Tomorrow*

---

## How to Use This Guide

Each section below corresponds to one `.pptx` file in your folder.  
For every topic you will find:

1. **What it is** — simple explanation for a newbie
2. **Analogy** — everyday comparison to build intuition
3. **Real-Life Use Case** — where it's used in industry
4. **Viva Tip** — likely examiner questions and short answers

---

# 📘 File 1: ML-Introduction.pptx
## What is Machine Learning?

### What it is
Machine Learning (ML) is a sub-field of Artificial Intelligence. Instead of writing explicit rules (if-then-else), you give the computer lots of examples (data) and let it figure out the rules itself. The computer improves its performance on a task through experience.

**Tom Mitchell's Definition:**  
*"A computer program is said to learn from experience E with respect to task T and performance measure P, if its performance on T, as measured by P, improves with experience E."*

### Analogy
Teaching a 5-year-old what a "dog" is. You don't give a biological definition. You show 100 photos labeled "dog" and 100 photos labeled "not dog." After enough examples, the child can identify a new animal correctly — even though you never explained what a snout or fur is.

### Real-Life Use Case
**Email Spam Filter:** You click "Report Spam" on junk emails. The system watches your behavior and learns patterns (words like "free money," "winner," strange sender addresses) so it can automatically filter future spam without you writing rules manually.

### Key Terms
| Term | Meaning |
|------|---------|
| Training Data | The examples you show the computer |
| Model | What the computer learns after studying the data |
| Prediction | When the model looks at new data and gives an answer |
| Features | Details the computer looks at (e.g., in emails: words, sender) |
| Labels | The correct answer (e.g., spam / not spam) |

### Viva Tip
**Q: What is Machine Learning?**  
**A:** Machine Learning is a field of AI where a computer learns patterns from data and improves its performance on a task without being explicitly programmed with rules.

---

# 📘 File 2: ML-1-part1-1.pptx
## Data Basics: Features, Vectors & Preprocessing

### Part A: Features, Vectors & Labels
**What it is:** To make a decision, you need details about something. Those details are **features**. A list of features together makes a **vector**. The answer you're trying to predict is the **label**.

**Analogy:** Buying a used car. Features: number of wheels, height, weight, color. Vector: `[4, 5.5, 600, Red]`. Label: "Sedan," "SUV," or "Truck."

### Part B: Data Preprocessing
Real-world data is messy — missing values, different scales, text categories. You must clean it before the computer can use it.

**Analogy:** Baking a cake. You have flour in cups, sugar in grams, and eggs counted individually. If you dump them together without converting to the same unit, the cake fails. Preprocessing is converting everything to the same measuring cup.

**Specific Techniques:**

1. **Normalization (Z-score & Min-Max):** Making all numbers fit on the same scale so one big number doesn't drown out small ones.  
   *Analogy:* Comparing height (in feet) and weight (in pounds) directly is unfair unless you scale them.

2. **Imputation:** Filling in missing data (e.g., replacing a missing age with the average age).

3. **Binning:** Grouping continuous numbers into categories (ages 0–12 = "Child", 13–19 = "Teen").

### Part C: Encoding Categories
Computers only understand numbers. Categories like "Red," "Blue," "Green" must be turned into numbers.

| Method | How it works | Best for |
|--------|--------------|----------|
| **Label Encoding** | Red=0, Blue=1, Green=2 | Ordered data (Low/Medium/High) |
| **One-Hot Encoding** | Separate binary columns: IsRed, IsBlue, IsGreen | Unordered data (colors, genres) |

**Problem with Label Encoding:** The computer thinks Green > Blue > Red (an order that doesn't exist). One-Hot Encoding avoids this fake ranking.

### Real-Life Use Case
A movie recommendation system. Genres (Action, Comedy, Drama) have no natural order, so you use One-Hot Encoding so the computer doesn't think "Action < Comedy < Drama."

### Viva Tip
**Q: Difference between Label Encoding and One-Hot Encoding?**  
**A:** Label Encoding assigns each category a number (good for ordered data like Low/Medium/High). One-Hot Encoding creates separate binary columns for each category (best for unordered data like colors or genres, because it avoids creating fake rankings).

---

# 📘 File 3: Regression 1.pptx
## Linear Regression — Predicting Numbers

### What it is
Regression predicts a **number** (price, temperature, salary) based on input features. Linear Regression assumes a straight-line relationship between input and output:  
`Price = (slope × Size) + intercept`

### Analogy
Guessing the price of a house based only on its size (square feet). You collect data on 100 houses: size and actual selling price. You draw the straight line that best fits all points. That line is your model. To predict a new house's price, find its size on the line and read the price.

### How the computer finds the best line
- **Least Squares Method:** The computer tries many lines and picks the one where the total distance from all points to the line is smallest.
- **Gradient Descent:** Imagine walking down a foggy mountain, taking small steps downhill until you reach the bottom (lowest error). α is the learning rate (step size).

### Real-Life Use Case
Predicting house prices, stock prices, or a student's exam score based on hours studied.

### Viva Tip
**Q: What is Linear Regression used for?**  
**A:** It predicts a continuous numerical value (like price or temperature) by finding the best straight-line relationship between input features and the output.

---

# 📘 File 4: Logistic Regression and LDA.pptx
## Logistic Regression — Predicting Categories

### What it is
Despite the name "Regression," this is for **Classification** — predicting a category (Yes/No, Spam/Not Spam, Sick/Healthy). Instead of predicting a number, it predicts a **probability** between 0 and 1 using the **Sigmoid Function** (squashes any number into 0–1 range).

### Analogy
Linear regression predicts "how sick is this patient?" (score 0–100). Logistic regression predicts "is this patient sick?" (Yes/No) by giving a probability like 85%. If probability > 50%, we say "Yes."

### Real-Life Use Case
Credit card fraud detection: Is this transaction fraudulent? (Yes/No). Or: Will this customer buy the product? (Yes/No).

### LDA — Linear Discriminant Analysis
**What it is:** Another classification method. It assumes data from each class forms a bell curve (Gaussian distribution) and draws a line to separate classes based on their averages.

**Analogy:** Two groups: tall basketball players and short gymnasts. You measure everyone's height. LDA finds the height that best separates the groups (e.g., 5'10"). Taller → basketball player; shorter → gymnast.

### LDA vs QDA
| | LDA | QDA |
|---|-----|-----|
| Assumption | Same spread (variance) for all classes | Different spread per class |
| Boundary | Straight line (linear) | Curved line (quadratic) |
| Data needs | Less data, lower variance | More data, higher flexibility |

### Viva Tip
**Q: How is Logistic Regression different from Linear Regression?**  
**A:** Linear Regression predicts a continuous number (house price). Logistic Regression predicts a probability for a category (spam vs not spam), using a sigmoid function to squash output between 0 and 1.

---

# 📘 File 5: Performance measures-ML-1-part2 1.pptx
## How do we know if our model is good?

### What it is
After building a model, test it on data it has never seen. Compare predictions to actual answers.

### Confusion Matrix (The Report Card)
For a Yes/No model:

| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | TP (True Positive) | FN (False Negative) |
| **Actual No** | FP (False Positive) | TN (True Negative) |

**Analogy:** A smoke detector:
- **TP:** Fire is happening, alarm goes off. (Correct!)
- **TN:** No fire, alarm silent. (Correct!)
- **FP:** No fire, but alarm goes off. (False alarm — annoying but safe)
- **FN:** Fire happening, but alarm silent. **Dangerous!**

### Key Metrics
| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP + TN) / Total | How often is the model right overall? |
| **Precision** | TP / (TP + FP) | When model says Yes, how often is it actually Yes? (Avoid false alarms) |
| **Recall (Sensitivity)** | TP / (TP + FN) | Of all actual Yes cases, how many did the model catch? (Avoid missing real cases) |
| **F1-Score** | Harmonic mean of Precision & Recall | Balances both Precision and Recall |
| **ROC-AUC** | Area under ROC curve | Overall performance; higher = better model |

### Real-Life Use Case
- **Cancer screening:** Recall matters most. You don't want to miss a real cancer case (FN is deadly), even if healthy people get false alarms (FP).
- **Spam filter:** Precision matters most. You don't want an important email marked as spam (FP), even if a few spam emails slip through (FN).

### Viva Tip
**Q: Difference between Precision and Recall?**  
**A:** Precision is about not crying wolf (when you predict positive, you're usually right). Recall is about not missing anything (you catch most actual positives). For cancer detection, recall matters more. For spam filtering, precision matters more.

---

# 📘 File 6: Bias-Variance trade-off 1.pptx
## Underfitting vs Overfitting

### What it is
- **Underfitting:** Model is too simple. Doesn't learn the pattern. Bad on training data AND bad on new data.
- **Overfitting:** Model memorizes training data (including mistakes/noise). Great on training data, terrible on new data.
- **Just Right:** Good balance — learns the pattern without memorizing noise.

### Analogy
Preparing for a math exam by practicing past papers:
- **Underfit:** You only memorize that 2+2=4. Can't solve anything new.
- **Just Right:** You understand concepts, so you can solve new problems.
- **Overfit:** You memorize exact answers to last year's paper. 100% on that paper, but fail the actual exam because questions are different.

### Regularization (The Fix)
When a model is too complex (overfitting), add a penalty to keep it simpler:
- **L1 Regularization (Lasso):** Can completely remove unimportant features (sets weight to 0). Used for feature selection.
- **L2 Regularization (Ridge):** Shrinks large weights but doesn't zero them. Used when features are correlated.
- **λ (lambda):** Large λ → underfit. Small λ → overfit.

### Analogy
Regularization is like a teacher saying: "Don't memorize every example; understand the general rule instead."

### Viva Tip
**Q: What is Overfitting?**  
**A:** Overfitting happens when a model learns the training data too well, including its noise and random fluctuations, so it performs great on training data but poorly on new, unseen data.

---

# 📘 File 7: Intro to Decision Tree.pptx
## Decision Trees

### What it is
A Decision Tree is like a flowchart of yes/no questions. You start at the top (root node), answer questions at internal nodes, and reach a bottom answer (leaf node).

### Analogy
Deciding whether to play outside:
1. Is it raining?  
   - Yes → Don't play.  
   - No → Next question.
2. Is temperature above 20°C?  
   - Yes → Play!  
   - No → Don't play.

Each question splits the data into two branches. The computer finds the best questions automatically.

### Real-Life Use Case
A bank deciding whether to approve a loan:
1. Is income above ₹50,000?
2. Is credit score above 700?
3. Has the person defaulted before?
Final leaf: Approve or Reject.

### How the tree splits: Entropy & Information Gain
The tree wants each split to make groups more "pure" (all one answer).
- **Entropy:** Measures how mixed up a group is. High entropy = chaos (50% Yes, 50% No). Low entropy = pure (100% Yes).
- **Information Gain:** How much chaos decreases after a split. Pick the split with highest gain.

**Example from your file:** Does "High Discount" make someone buy?  
Before split: 5 Yes, 5 No (chaos = 1). After split: High Discount → mostly Yes; Low Discount → mostly No. Information Gain = 0.61 (good split!).

### Pruning (Preventing Overfitting)
Trees can grow too big and memorize noise. Pruning cuts off useless branches:
- **Pre-pruning:** Stop growing early (limit depth, min samples).
- **Post-pruning:** Grow full tree, then cut useless branches.
- **Cost-Complexity Pruning (α):** Balances tree simplicity vs accuracy. Small α = complex tree (overfit risk). Large α = simple tree (underfit risk).

### Viva Tip
**Q: How does a Decision Tree decide where to split?**  
**A:** It uses a greedy approach to find the split that makes resulting groups most homogeneous (purest). For classification, it uses Entropy or Information Gain (or Gini Index) to pick the best feature and split point.

---

# 📘 File 8: K-means Clustering 1.pptx
## K-Means Clustering (Unsupervised Learning)

### What it is
Clustering groups similar things together when you have **no labels**. K-Means divides data into K groups.

### Analogy
You have 100 unlabeled fruits in a pile. You want to group them. You pick 3 random spots on the table (centroids). Each fruit goes to the nearest spot. Then move each spot to the average location of its fruits. Repeat until spots stop moving. You now have 3 piles — probably apples, oranges, bananas.

### Real-Life Use Case
**Customer segmentation:** A supermarket groups customers by spending habits: "budget shoppers," "premium shoppers," "occasional shoppers." Then sends different coupons to each group.

### Choosing K: The Elbow Method
Try K = 1, 2, 3... Calculate WCSS (Within-Cluster Sum of Squares = how far points are from their group center). Plot WCSS vs K. The curve drops sharply then flattens — the "elbow" is the optimal K.

### Performance Measures
| Measure | Range | Meaning |
|---------|-------|---------|
| **Silhouette Score** | -1 to 1 | 1 = point well-matched to its group and far from other groups |
| **Davies-Bouldin Index** | Lower is better | Average similarity between each cluster and its most similar cluster |
| **Calinski-Harabasz (CH) Score** | Higher is better | Ratio of between-cluster dispersion to within-cluster dispersion |

### Limitations of K-Means
- Must choose K beforehand
- Assumes round/spherical clusters of similar size
- Sensitive to initial centroid placement (use k-means++)
- Sensitive to outliers (one weird point pulls the center)
- Not good for high-dimensional data (curse of dimensionality)

### Viva Tip
**Q: What is the Elbow Method in K-Means?**  
**A:** Run K-Means for different K values and plot WCSS against K. The "elbow" point — where adding more clusters stops giving much improvement — is chosen as the optimal number of clusters.

---

# 📘 File 9: Clustering.pptx
## Other Clustering Methods

### DBSCAN (Density-Based Clustering)
**What it is:** Groups points packed closely together; marks loners as noise. Doesn't need K specified.

**Analogy:** People at a party. People in tight clusters are "groups." A person standing alone in the corner is "noise."

**Key Parameters:**
- **ε (epsilon):** How close is "close"?
- **MinPts:** How many points needed to form a group?

**Point Types:**
- **Core point:** Has at least MinPts neighbors within ε. Forms cluster center.
- **Border point:** Not a core point, but within ε of a core point. Belongs to cluster but doesn't expand it.
- **Noise/Outlier:** Not within ε of any core point. Unclustered.

**Why better than K-Means sometimes:**
- No need to specify K
- Finds arbitrary-shaped clusters (not just circles)
- Handles outliers gracefully

### Hierarchical Clustering (Connectivity-Based)
**What it is:** Builds a tree of groups (dendrogram).

| Type | Approach | How it works |
|------|----------|--------------|
| **Agglomerative** | Bottom-up | Each point starts alone. Merge closest two. Repeat until one group. |
| **Divisive** | Top-down | Start with all in one group. Split into two. Repeat until each point alone. |

**Analogy:** A family tree. You and your sibling merge first, then with cousins, then extended family. The dendrogram shows exactly when each merge happened.

### Linkage Methods (Distance between groups)
| Method | Definition | Best for |
|--------|------------|----------|
| **Single** | Shortest distance between any two members of different clusters | Long, snake-like shapes; geospatial |
| **Complete** | Longest distance between any two members | Tight, round clusters |
| **Average** | Average distance between all members | General purpose |
| **Ward's** | Minimizes variance increase when merging | Compact, equal-sized clusters |

### Real-Life Use Case
- **Document clustering:** Grouping news articles by topic.
- **Gene expression:** Finding genes with similar behavior.
- **Customer segmentation:** Grouping customers with similar buying behavior.

### Viva Tip
**Q: Difference between K-Means and DBSCAN?**  
**A:** K-Means requires you to specify the number of clusters (K) and assumes clusters are spherical. DBSCAN does not require K, can find clusters of arbitrary shape, and identifies noise/outliers as points that don't belong to any cluster.

---

# 📘 File 10: ANN & Perceptron.pptx
## Neural Networks & The Perceptron

### What it is
Artificial Neural Networks (ANN) are inspired by the human brain. They have neurons connected together, organized in layers (input → hidden → output). Information flows in one direction (Feedforward ANN).

### Analogy
A team of people passing notes. Each person (neuron) receives notes, does a simple calculation, and passes a note to the next person. The last person gives the final answer.

### The Perceptron
The simplest neuron (introduced by Frank Rosenblatt, 1957). Takes inputs, multiplies by weights, adds them up, and says "+1" or "-1" based on whether the weighted sum is ≥ 0. Used for **binary classification** of **linearly separable** patterns.

**Real-Life Use Case:** Simple binary decisions — Is this banknote authentic? (Yes/No) based on features like width, color intensity.

### The Big Limitation: XOR
A single-layer perceptron can only draw straight lines to separate data. **XOR** needs a curved boundary — impossible for one layer.

**Analogy:** You can't draw one straight line on a table to separate four coins arranged in an XOR pattern.

### Multilayer Perceptron (MLP)
Adds hidden layers between input and output. Allows curved, complex boundaries. But because hidden layer targets are unknown, normal learning fails — you need **Back-Propagation**.

### Viva Tip
**Q: Why can't a single-layer Perceptron solve XOR?**  
**A:** XOR is not linearly separable — there is no single straight line that can separate the cases correctly. A single-layer perceptron only learns linear decision boundaries, so it fails on XOR.

---

# 📘 File 11: Support vector machine (SVM) 2.pptx
## Support Vector Machines

### What it is
SVM finds the best boundary (hyperplane) to separate two classes, with the widest possible gap (margin) between them.

**Analogy:** Two rival gangs on a field. Build a fence exactly in the middle, as far as possible from both gangs. The closest members of each gang to the fence are **Support Vectors** — they "hold up" the fence. If they move, the fence moves.

### Hyperplane
In p-dimensional space, a hyperplane is a flat subspace of dimension p-1 (line in 2D, plane in 3D).

### Hard vs Soft Margin
| | Hard Margin | Soft Margin (SVM Classifier) |
|---|-------------|------------------------------|
| Mistakes allowed | None | Some allowed |
| Best when | Data perfectly separable | Real-world messy data |
| Parameter | None | **C** (regularization) |

**Role of C:**
- **High C:** Strict about mistakes. Narrow margin. Risks overfitting.
- **Low C:** Tolerant of mistakes. Wide margin. Simpler boundary.

### What if classes can't be separated by a straight line? → The Kernel Trick
**What it is:** If you can't separate points on a flat table, lift them into 3D. Suddenly they separate easily. The Kernel Trick does this mathematically without actually working in higher dimensions.

**Analogy:** Two seed types in concentric circles on a plate. You can't draw a line to separate them. Shake the plate — seeds jump into the air — now you can slice horizontally between them. The kernel is the "shake."

**Common Kernels:**
- **Linear:** Straight line boundary
- **Polynomial:** Curved boundary (choose degree d)
- **RBF (Radial Basis Function):** Very flexible, can make complex closed shapes. Most popular.

### Multi-class SVM
- **One-vs-One:** Train an SVM for every pair of classes.
- **One-vs-All:** Train K SVMs, each comparing one class against all others.

### Real-Life Use Case
Image recognition (face detection), text classification (spam vs not spam), bioinformatics (protein classification).

### Viva Tip
**Q: What is the Kernel Trick in SVM?**  
**A:** It transforms data into a higher dimension where it becomes linearly separable, without explicitly computing the transformation. It uses a kernel function to compute similarity between points in the original space. Common kernels: Linear, Polynomial, RBF.

---

# 📘 File 12: After mid-term.pptx
## Advanced Topics: Boosting & Stacking

This file covers advanced ensemble methods.

### Boosting
**What it is:** Instead of one strong model, combine many **weak models** one after another. Each new model fixes the mistakes of the previous one.

**Analogy:** Learning to shoot arrows at a target:
- Shot 1: Miss to the left.
- Shot 2: Aim more right to fix the left miss. Now miss slightly high.
- Shot 3: Aim lower to fix height.
- Final: Combine all three adjustments to hit the bullseye.

### AdaBoost (Adaptive Boosting)
1. Start with equal importance (weights) for all data points.
2. Train a model.
3. Points it got wrong get higher importance.
4. Train next model focusing on hard points.
5. Combine all models with weighted vote.

**Real-Life Use Case:** AdaBoost was the first successful boosting algorithm for binary classification. It combines multiple weak classifiers into one strong classifier.

### Other Boosting Algorithms
| Algorithm | Key Idea |
|-----------|----------|
| **Gradient Boosting** | Each new model predicts the errors (residuals) of the previous one using gradient descent |
| **XGBoost** | Optimized Gradient Boosting with regularization; faster; handles numerical + categorical |
| **CatBoost** | Great for categorical features without preprocessing |
| **SMOTEBoost** | Combines boosting with oversampling of minority class |
| **RUSBoost** | Combines boosting with undersampling of majority class |

### Stacking Ensemble Model
**What it is:** A team of specialists consults a judge.

- **Base Models (Level-1):** Different models (Decision Tree, SVM, Neural Net, Random Forest) each make predictions.
- **Meta-Learner (Level-2):** A final model that sees all base models' predictions and makes the final decision.

**Analogy:** You're sick. You consult a cardiologist, neurologist, and general physician. Each gives an opinion. A senior doctor (meta-learner) looks at all three opinions and makes the final diagnosis.

**Steps:**
1. Train multiple base models on same data.
2. Generate predictions from base models.
3. Use predictions as input for meta-learner.
4. Train meta-learner on new dataset (features = base predictions, target = actual labels).
5. Final prediction on test data: base models predict → meta-learner decides.

### Real-Life Use Case
Kaggle competitions — winners almost always use stacking or boosting because it combines strengths of different models.

### Bagging vs Boosting
| | Bagging | Boosting |
|---|---------|----------|
| Training | Parallel, independent | Sequential, dependent |
| Data | Bootstrap samples | Full data with re-weighted errors |
| Goal | Reduce variance | Reduce bias and variance |
| Example | Random Forest | AdaBoost, XGBoost |

### Viva Tip
**Q: Difference between Bagging and Boosting?**  
**A:** Bagging trains many models independently on different data samples and averages them (reduces variance, good for overfitting). Boosting trains models sequentially, where each new model focuses on fixing errors of previous ones (reduces both bias and variance).

---

# 📘 File 13 & 14: Resampling Methods + Decision Trees + Random Forest + Boosting

These two files cover the same topics with slight duplication. Here is the unified explanation.

### Resampling Methods (Checking if your model is reliable)

**What it is:** You only have one dataset. How do you know your model works on new data? Simulate multiple datasets by resampling.

| Method | How it works | Pros | Cons |
|--------|--------------|------|------|
| **Hold-Out Validation** | Split Train (80%) / Test (20%) | Simple | Result depends on which 20%; trains on less data |
| **LOOCV** | Use 1 point as test, train on rest. Repeat for every point. | Very unbiased; uses almost all data | Extremely slow (train N times for N points) |
| **K-Fold CV** | Split into K=5 or 10 folds. Train on K-1, test on 1. Repeat K times. | Reliable; good bias-variance balance | More computation than hold-out |
| **Stratified K-Fold** | Like K-Fold, but each fold keeps same class proportion | Great for imbalanced data | Slightly more complex |

**Analogy:** Instead of one final exam, take 10 practice tests, each time leaving out a different chapter. Your average score is more reliable than one test.

**Bias-Variance Trade-off in CV:**
- **LOOCV:** Low bias, high variance (training sets are almost identical, so results are correlated)
- **K-Fold (k=5 or 10):** Intermediate bias and variance — empirically the best choice

### Decision Trees (Recap)
Same as File 7:
- Root node → internal nodes → leaf nodes
- Recursive binary splitting (greedy)
- Entropy / Information Gain for classification splits
- Pruning (pre-pruning, post-pruning, cost-complexity pruning with α)

### Random Forest
**What it is:** A forest of decision trees. Each tree trained on a random subset of data **and** a random subset of features. Majority vote (classification) or average (regression).

**Analogy:** Buying a car. Instead of asking one friend, ask 100 friends. Each friend knows only some features (price, mileage, safety). Each gives a recommendation. Majority choice wins.

**Why better than one tree:**
- One tree overfits (high variance).
- Random Forest averages out noise — more stable and accurate.
- Great for high-dimensional data (many features).

**Two sources of randomness:**
1. **Bootstrap sampling:** Each tree on different data sample.
2. **Feature randomness:** Each tree considers only random subset of features when splitting.

### Viva Tip
**Q: What is Random Forest, and how is it different from a single Decision Tree?**  
**A:** Random Forest is an ensemble of many decision trees. Each tree is trained on a bootstrap sample of the data and a random subset of features. It reduces overfitting (high variance of a single tree) by averaging many trees, making it more accurate and robust than a single decision tree.

---

# 🎯 Quick Revision Cheat Sheet for Viva

| If they ask... | Say this in 1 line |
|---|---|
| What is ML? | Learning patterns from data instead of explicit rules. |
| Supervised vs Unsupervised? | Supervised has labels (answers); unsupervised finds hidden patterns without labels. |
| Regression vs Classification? | Regression predicts numbers; Classification predicts categories. |
| Overfitting? | Memorizing training data; fails on new data. Fix with regularization or pruning. |
| K-Means vs DBSCAN? | K-Means needs K and round clusters; DBSCAN finds any shape and auto-detects noise. |
| Decision Tree split? | Best split = highest Information Gain / lowest Entropy (most pure groups). |
| SVM Kernel? | Trick to separate non-linear data by mapping to higher dimensions. |
| Bagging vs Boosting? | Bagging = parallel, independent trees averaged. Boosting = sequential, fixing errors. |
| Precision vs Recall? | Precision = don't cry wolf. Recall = don't miss real cases. |
| Cross-Validation? | Testing model on different subsets to get a reliable performance estimate. |
| Random Forest? | Many decision trees voting together, each on random data and random features. |

---

## 📋 Final Exam-Day Checklist

- [ ] Can explain ML with a real-life example (spam filter, house price, etc.)
- [ ] Can draw and explain a Confusion Matrix
- [ ] Can explain the difference between Precision and Recall with examples
- [ ] Can explain Overfitting vs Underfitting with the exam-preparation analogy
- [ ] Can describe how K-Means works and how to choose K (Elbow method)
- [ ] Can explain Decision Tree splitting (Entropy/Information Gain) and pruning
- [ ] Can explain SVM, Support Vectors, and the Kernel Trick simply
- [ ] Can explain Boosting vs Bagging with analogies
- [ ] Can explain Random Forest and why it's better than one tree
- [ ] Can explain Cross-Validation (K-Fold, LOOCV, Stratified)

**Good luck with your viva!** Focus on understanding the intuition (the analogies) rather than memorizing formulas — examiners love it when you can explain with a real-life example.
