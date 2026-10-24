"""Train and compare simple models for the UCI German Credit dataset."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "german_credit.data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

COLUMN_NAMES = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "credit_risk",
]

NUMERIC_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load raw data and make 1 mean bad credit risk (the positive class)."""
    data = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES)
    features = data.drop(columns="credit_risk")
    target = data["credit_risk"].map({1: 0, 2: 1})
    return features, target


def make_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    """Prepare each feature type without using any test-set information."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, pd.Series]:
    """Measure a fitted model on applicants it did not see during training."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    return metrics, probabilities


def save_charts(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, probabilities, model_name: str) -> None:
    """Create the two visual checks requested in the internship task."""
    predictions = model.predict(x_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["Good risk", "Bad risk"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    score = roc_auc_score(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(false_positive_rate, true_positive_rate, label=f"{model_name} (AUC = {score:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve", xlim=(0, 1), ylim=(0, 1.05))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the complete beginner-friendly training workflow."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    features, target = load_data()
    categorical_features = [column for column in features.columns if column not in NUMERIC_FEATURES]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    }

    results = []
    fitted_models = {}
    probabilities_by_model = {}
    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(categorical_features)),
                ("model", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        metrics, probabilities = evaluate_model(name, pipeline, x_test, y_test)
        results.append(metrics)
        fitted_models[name] = pipeline
        probabilities_by_model[name] = probabilities

    results_frame = pd.DataFrame(results).sort_values(["f1_score", "roc_auc"], ascending=False)
    results_frame.to_csv(OUTPUT_DIR / "model_results.csv", index=False, float_format="%.3f")

    best_name = results_frame.iloc[0]["model"]
    best_model = fitted_models[best_name]
    joblib.dump(best_model, OUTPUT_DIR / "best_credit_model.joblib")
    save_charts(best_model, x_test, y_test, probabilities_by_model[best_name], best_name)

    print("Dataset rows:", len(features))
    print("Training rows:", len(x_train))
    print("Test rows:", len(x_test))
    print("\nTest-set results (1 means bad credit risk):")
    print(results_frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected model: {best_name}")
    print(f"Saved results and charts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
