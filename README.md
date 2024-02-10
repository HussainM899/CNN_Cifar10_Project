# CNN CIFAR-10 Image Classification

## Project Overview
This repository contains a Convolutional Neural Network (CNN) designed to classify images from the CIFAR-10 dataset. The CNN is implemented using Keras with TensorFlow as the backend.

## Introduction
The CIFAR-10 dataset includes 60,000 32x32 color images across 10 different classes. The objective of this project is to develop a CNN model that classifies these images with a high degree of accuracy.

## Dataset
The CIFAR-10 dataset comprises 50,000 training images and 10,000 testing images spread across 10 classes, which include vehicles and animals.

## Model Architecture
The model is built using a Sequential approach in Keras and consists of several layers designed to process and classify images:

1. Convolutional layers for extracting features
2. MaxPooling layers for dimensionality reduction
3. A Flatten layer to convert the 2D outputs to 1D
4. Dense layers for final classification

## Training
The model is trained using the following specifications:
- Optimizer: RMSprop
- Loss function: Categorical crossentropy
- Metric: Accuracy
- Epochs: 10

## Evaluation
Evaluation is performed using the test set of CIFAR-10 with metrics including accuracy, precision, recall, F1-score, and a confusion matrix to assess the model's performance.

## Results
The trained model was able to reach an accuracy of ~69% on the test dataset. For more detailed performance metrics, please refer to the evaluation section in the Jupyter notebook provided in this repository.

## Improvement Strategies
To further improve the model, you might consider:
- Applying data augmentation techniques to enrich the dataset
- Tuning hyperparameters like learning rate, batch size, or layer configurations
- Experimenting with different model architectures, such as ResNet or Inception

## Usage
To train the model and evaluate its performance, execute the training script:

```bash
python train.py
```

## Contribution
Contributions to this project are welcome! Please feel free to fork the repository, make your changes, and create a pull request.


## Dependencies
- Python 3.6 or above
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn

## Installation
Clone the project repository and install the necessary packages using:

```bash
git clone https://github.com/<HussainM889>/CNN_CIFAR10.git
cd CNN_CIFAR10
pip install -r requirements.txt
