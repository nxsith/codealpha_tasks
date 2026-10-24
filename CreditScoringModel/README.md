# Credit Scoring Model

This is a beginner-friendly machine learning project for CodeAlpha Task 1. It predicts whether a credit applicant belongs to a good-credit or bad-credit-risk category using past financial information.

> Educational project only. It must not be used to make real lending decisions.

## What the project will do

1. Load the UCI German Credit dataset.
2. Prepare numerical and category-based information for machine learning.
3. Train Logistic Regression and Random Forest models.
4. Compare Accuracy, Precision, Recall, F1-score, and ROC-AUC.
5. Save a confusion matrix, ROC curve, results table, and the better model.

## Folder guide

- `data/` holds the original dataset.
- `outputs/` holds the model and charts created after training.
- `train_credit_model.py` contains the complete training process.

## Concepts we will learn

- **Feature:** an input column, such as loan duration or savings status.
- **Target:** the answer to learn, here credit risk.
- **Training data:** examples the model is allowed to learn from.
- **Test data:** examples held back until the end to check whether the model works on unseen applicants.
- **Classification:** choosing one category from a fixed set; here, good or bad credit risk.

## How to run

From this project folder, create an environment and install the required libraries:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_credit_model.py
```

After training, look in `outputs/` for:

- `model_results.csv` - the comparison table for both models.
- `confusion_matrix.png` - correct and incorrect classifications for the selected model.
- `roc_curve.png` - the selected model's ability to separate good and bad risk across thresholds.
- `best_credit_model.joblib` - the saved selected model.

## Dataset

The project uses the [UCI German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), which contains 1,000 historical credit applicants. The target is converted to:

- `0`: good credit risk
- `1`: bad credit risk

The input features include loan duration, loan amount, checking and savings status, employment duration, age, housing, and other historical credit information.

## Preprocessing

The dataset has two kinds of inputs:

- **Numerical features**, such as age and credit amount. These are scaled so very large-number columns do not dominate the model.
- **Categorical features**, such as housing or savings status. These are converted into separate yes/no columns with one-hot encoding.

The preprocessing is inside a scikit-learn pipeline. This is important: it learns settings from the training data only, which avoids accidentally using information from the test set.

## Model results

We kept 20% of the data (200 applicants) as an unseen test set. The remaining 80% (800 applicants) was used for training.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.780 | 0.667 | 0.533 | 0.593 | 0.804 |
| Random Forest | 0.750 | 0.583 | 0.583 | 0.583 | 0.801 |

Logistic Regression was selected because it had the higher F1-score and ROC-AUC on the held-out test data.

## Understanding the measurements

- **Accuracy:** the proportion of all predictions that were correct.
- **Precision:** when the model predicts bad risk, how often that prediction was correct.
- **Recall:** of all historical bad-risk applicants, how many the model found.
- **F1-score:** a balance between precision and recall.
- **ROC-AUC:** how well the model separates the two groups across many decision thresholds; 0.5 is similar to random guessing, while 1.0 is perfect separation.

## Limitation and responsible use

This model is an educational demonstration using historical data. It is not suitable for real lending or approval decisions. In particular, its recall shows it misses some applicants historically labelled as bad risk, and the dataset itself may not represent current applicants or fair lending practice.
