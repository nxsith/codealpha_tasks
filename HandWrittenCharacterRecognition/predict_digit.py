import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "outputs" / "digit_cnn.keras"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one handwritten digit image")
    parser.add_argument("--image", required=True, help="Path to a digit image")
    return parser.parse_args()


def prepare_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("L").resize((28, 28))
    pixels = np.asarray(image, dtype="float32") / 255.0
    if pixels.mean() > 0.5:
        pixels = 1.0 - pixels
    return pixels.reshape(1, 28, 28, 1)


def main() -> None:
    args = parse_arguments()
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: python train_cnn.py")

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(prepare_image(args.image), verbose=0)[0]
    digit = int(np.argmax(probabilities))
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {probabilities[digit]:.1%}")


if __name__ == "__main__":
    main()
