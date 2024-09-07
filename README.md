# Covid19-detection-through-Tensorflow
## CNN Model for COVID-19 Image Classification

This project implements a Convolutional Neural Network (CNN) model to classify medical images as either COVID-19 positive or normal. The project uses Keras and TensorFlow to build and train the model, with the option to classify new images using a saved model.

# Key Features
## Model Architecture: The CNN consists of multiple layers, including 2D Convolutional layers, MaxPooling, Batch Normalization, Dropout for regularization, and Dense layers for classification.

## Convolution Layers: Five convolutional layers with different filter sizes (32, 64, 96) for extracting spatial features from the images.

## Pooling Layers: MaxPooling layers are used after each convolutional layer to reduce the dimensionality and focus on important features.

## Batch Normalization: This helps in stabilizing the learning process and improves convergence speed.

## Dropout Regularization: Reduces overfitting by randomly dropping some connections during training.

## Dense Layers: Two fully connected layers at the end, with softmax activation for classification into two categories: COVID and Normal.

## Model Saving and Loading: After training, the model is saved as a JSON file (model.json) along with the weights (model.weights.h5) for future classification tasks.A separate classification function loads this saved model and predicts the label of new images.
