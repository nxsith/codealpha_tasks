# Dataset: MNIST Handwritten Digits

This project uses the MNIST dataset of handwritten digits 0-9. TensorFlow downloads it automatically the first time `train_cnn.py` runs.

- Training images: 60,000
- Test images: 10,000
- Image shape: 28 x 28 grayscale pixels
- Classes: 10 (`0` through `9`)

Pixel values are converted from 0-255 into 0-1 before training. The test set stays separate from model training and is used only for the final evaluation.
