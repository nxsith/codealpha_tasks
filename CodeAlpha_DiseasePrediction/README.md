# Disease Prediction From Medical Data

This beginner-friendly project is for CodeAlpha Task 4. It compares machine-learning models that classify the historical Pima Indians Diabetes dataset into negative (`0`) and positive (`1`) outcomes.

> Educational exercise only. This model is not a medical device and must not be used for diagnosis, treatment, or medical decision-making.

## What the project does

1. Loads one structured medical dataset with eight patient features.
2. Converts impossible zero values in selected measurements into missing values.
3. Imputes missing values and scales the features in a safe preprocessing pipeline.
4. Trains Logistic Regression, Random Forest, and SVM classifiers.
5. Compares Accuracy, Precision, Recall, F1-score, and ROC-AUC on unseen test data.
6. Saves the selected model, a confusion matrix, and an ROC curve.

## Dataset

The project uses the Pima Indians Diabetes dataset, a well-known educational dataset associated with the UCI Machine Learning Repository. It contains 768 historical records, eight input features, and the `Outcome` label.

| Feature | Meaning |
|---|---|
| Pregnancies | Number of pregnancies. |
| Glucose | Plasma glucose concentration. |
| BloodPressure | Diastolic blood pressure. |
| SkinThickness | Triceps skin-fold thickness. |
| Insulin | Two-hour serum insulin. |
| BMI | Body mass index. |
| DiabetesPedigreeFunction | A dataset-specific family-history score. |
| Age | Age in years. |

For `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`, a zero is not physiologically plausible. The training script treats zero as missing only in these columns and replaces it with a training-data median through the pipeline.

## Project structure

```text
CodeAlpha_DiseasePrediction/
├── data/
│   ├── diabetes.csv
│   └── README.md
├── outputs/
│   ├── best_diabetes_model.joblib
│   ├── confusion_matrix.png
│   ├── model_results.csv
│   └── roc_curve.png
├── train_disease_model.py
├── predict_diabetes.py
├── requirements.txt
└── README.md
```

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_disease_model.py
```

To make an educational example prediction after training:

```bash
python predict_diabetes.py \
  --pregnancies 2 --glucose 120 --blood-pressure 70 \
  --skin-thickness 30 --insulin 100 --bmi 28.5 \
  --pedigree 0.5 --age 35
```

## Final results

The data was split into 614 training records and 154 unseen test records. The table below reports test-set performance, where `1` is a positive dataset outcome.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.747 | 0.623 | 0.704 | 0.661 | 0.828 |
| SVM | 0.727 | 0.630 | 0.537 | 0.580 | 0.802 |
| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |

Random Forest was selected because it had the highest F1-score and ROC-AUC on the held-out test set.

## Understanding the selected model

On the 154 unseen test records, the Random Forest correctly classified 77 negative and 38 positive outcomes. It incorrectly labelled 23 negative outcomes as positive and missed 16 positive outcomes.

- **Recall (0.704):** it found about 70% of the positive outcomes in this test set.
- **Precision (0.623):** when it predicted a positive outcome, it was correct about 62% of the time.
- **ROC-AUC (0.828):** it separated the two historical outcome groups much better than random guessing, but not perfectly.

## Limitation and responsible use

The dataset represents a limited historical population and does not include all the information a clinician needs. A machine-learning output is not a diagnosis, and no person should make health decisions based on this project.
