"""Make one educational prediction with the saved diabetes model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "outputs" / "best_diabetes_model.joblib"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Educational diabetes dataset prediction")
    parser.add_argument("--pregnancies", type=float, required=True)
    parser.add_argument("--glucose", type=float, required=True)
    parser.add_argument("--blood-pressure", type=float, required=True)
    parser.add_argument("--skin-thickness", type=float, required=True)
    parser.add_argument("--insulin", type=float, required=True)
    parser.add_argument("--bmi", type=float, required=True)
    parser.add_argument("--pedigree", type=float, required=True)
    parser.add_argument("--age", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: python train_disease_model.py")

    args = parse_arguments()
    record = pd.DataFrame(
        [
            {
                "Pregnancies": args.pregnancies,
                "Glucose": args.glucose,
                "BloodPressure": args.blood_pressure,
                "SkinThickness": args.skin_thickness,
                "Insulin": args.insulin,
                "BMI": args.bmi,
                "DiabetesPedigreeFunction": args.pedigree,
                "Age": args.age,
            }
        ]
    )
    model = joblib.load(MODEL_PATH)
    probability = model.predict_proba(record)[0, 1]
    predicted_outcome = model.predict(record)[0]

    print("Educational dataset prediction only - not medical advice or a diagnosis.")
    print(f"Predicted dataset outcome: {'Positive' if predicted_outcome == 1 else 'Negative'}")
    print(f"Model probability of a positive dataset outcome: {probability:.1%}")


if __name__ == "__main__":
    main()
