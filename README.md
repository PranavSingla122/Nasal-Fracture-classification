# Nasal-Fracture-classification
Overview

This project aims to classify nasal fractures using deep learning models trained on X-ray images. The model is based on transfer learning with ResNet-34 and optimized for binary classification (normal vs. fractured nasal bones).

Dataset(https://github.com/IMobinI/Nasal-Bone-Fracture-Classification/tree/main/Dataset)

The dataset consists of X-ray images categorized into two classes:

Normal: Images of nasal bones without fractures.

Fractured: Images showing nasal fractures.

The dataset is loaded using PyTorch's ImageFolder and split into training and testing sets with an 80-20 ratio.
Model Architecture

Backbone Model: ResNet-34 (pretrained on ImageNet)

Fully Connected Layer: A single neuron with a sigmoid activation for binary classification

Loss Function: Binary Cross-Entropy with Logits

Optimizer: AdamW with weight decay

Scheduler: ReduceLROnPlateau to adapt the learning rate dynamically

Data Augmentation

The dataset is augmented using:

Random horizontal and vertical flips

Random rotation (30 degrees)

Shearing transformations

Random resized cropping

Color jittering (brightness, contrast, saturation, hue)

Gaussian blur

Random erasing

Training Process

Loads the dataset and applies transformations.

Uses transfer learning with ResNet-34.

Freezes early layers and fine-tunes deeper layers.

Applies early stopping based on validation loss.

Saves the best model checkpoint.

Evaluation Metrics

After training, the model is evaluated using:

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

ROC Curve and AUC-ROC Score

Optimal Decision Threshold Selection
Results Visualization

Loss curves for training and validation

Confusion matrix heatmap

AUC-ROC curve and score
