# Handwritten Digit Recognition with CNN

This beginner-friendly project is for CodeAlpha Task 3. It uses a Convolutional Neural Network (CNN) to recognise handwritten digits from the MNIST dataset.

> Educational project only. It recognises the standard MNIST digit format and is not a general handwriting or document-reading system.

## What the project does

1. Downloads and loads MNIST handwritten digits (`0` to `9`).
2. Normalises grayscale pixels from `0-255` to `0-1`.
3. Trains a small CNN for three epochs.
4. Evaluates it on 10,000 images never used for training.
5. Saves the trained model, learning curves, a confusion matrix, and sample predictions.
6. Predicts a digit from a supplied image file.

## Dataset

MNIST contains 60,000 training images and 10,000 test images. Every image is 28 by 28 pixels, and every image has one correct label from `0` through `9`.

## CNN architecture

```text
28 x 28 grayscale image
    ↓
Conv2D (32 filters) + ReLU
    ↓
MaxPooling2D
    ↓
Conv2D (64 filters) + ReLU
    ↓
MaxPooling2D
    ↓
Flatten → Dense (64 units) → Dropout → Dense (10 outputs, Softmax)
```

- **Conv2D** learns small visual patterns such as edges and curves.
- **ReLU** lets the model learn non-linear patterns.
- **MaxPooling2D** reduces the image representation while keeping the strongest patterns.
- **Flatten** converts the learned feature maps into one list.
- **Dense** layers combine the learned patterns to choose a digit.
- **Dropout** temporarily hides some training connections to reduce overfitting.
- **Softmax** turns the ten final scores into probabilities for digits 0-9.

## Project structure

```text
HandWrittenCharacterRecognition/
├── data/
│   └── README.md
├── outputs/
│   ├── digit_cnn.keras
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
├── train_cnn.py
├── predict_digit.py
├── requirements.txt
└── README.md
```

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_cnn.py
```

After training, predict an image containing one handwritten digit:

```bash
python predict_digit.py --image path/to/your_digit.png
```

The prediction script converts the image to grayscale, resizes it to 28 by 28 pixels, and inverts a light-background image to better match MNIST's light-digit-on-dark-background style.

## Final result

The model was trained for three epochs using a 10% validation split from the 60,000 training images. It achieved **98.79% test accuracy** on the 10,000 unseen MNIST test images.

The training and validation accuracy increased across the three epochs while their loss decreased. This is a positive sign that the model was learning useful patterns during this short training run.

## Limitation

MNIST is clean and standardised. Real handwriting may have different stroke widths, background colours, camera blur, multiple digits, or unusual writing styles. A high MNIST score does not prove that the model will work equally well on real-world handwriting.
