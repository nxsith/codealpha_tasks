from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def load_and_prepare_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, y_train, x_test, y_test


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_training_history(history: tf.keras.callbacks.History) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set(title="Training Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_history.png", dpi=150)
    plt.close(fig)


def save_confusion_matrix(y_test: np.ndarray, predictions: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=range(10), cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("MNIST Test-Set Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_sample_predictions(x_test: np.ndarray, y_test: np.ndarray, predictions: np.ndarray) -> None:
    sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for axis, index in zip(axes.flat, sample_indices):
        correct = predictions[index] == y_test[index]
        axis.imshow(x_test[index].squeeze(), cmap="gray")
        axis.set_title(
            f"True: {y_test[index]} | Pred: {predictions[index]}",
            color="green" if correct else "red",
            fontsize=9,
        )
        axis.axis("off")
    fig.suptitle("Sample MNIST Predictions", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    tf.keras.utils.set_random_seed(42)

    x_train, y_train, x_test, y_test = load_and_prepare_data()
    model = build_model()
    history = model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128, verbose=2)

    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    test_accuracy = accuracy_score(y_test, predictions)

    model.save(OUTPUT_DIR / "digit_cnn.keras")
    save_training_history(history)
    save_confusion_matrix(y_test, predictions)
    save_sample_predictions(x_test, y_test, predictions)

    print("Training images:", len(x_train))
    print("Test images:", len(x_test))
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved model and charts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
