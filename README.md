# Image Classification with CNN & Transfer Learning

This project was developed as part of the AI Bootcamp by the Saudi Digital Academy. It demonstrates the application of Convolutional Neural Networks (CNNs) and Transfer Learning techniques for image classification using the CIFAR-10 dataset.

## 📌 Project Overview

We explored and compared:
- A custom-built CNN architecture.
- Pre-trained CNN models (VGG16 and ResNet50) using Transfer Learning.

The goal was to classify images into 10 categories (e.g., airplane, automobile, cat, etc.) and evaluate the performance of different models.

## 🧠 Models Used
- **Custom CNN:** Built from scratch using Conv2D, MaxPooling, Dropout, and Dense layers.
- **VGG16:** A 16-layer pre-trained CNN model.
- **ResNet50:** A 50-layer deep residual network with skip connections.

## 🧪 Dataset
- **CIFAR-10**: 60,000 32x32 color images in 10 classes.
- Data preprocessing steps included normalization, one-hot encoding, and data augmentation (random rotations, flips, shifts).

## ⚙️ Training Techniques
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau

## ✅ Results

| Model        | Test Accuracy |
|--------------|----------------|
| Custom CNN   | 86.60%         |
| Custom CNN 2 | 91.58%         |
| **VGG16**    | **92.21%**     |
| ResNet50     | 91.31%         |

VGG16 achieved the highest accuracy and was selected as the best model due to its consistent and reliable performance.

## 🔍 Key Learnings
- Transfer Learning significantly boosts model performance with limited training.
- Data augmentation improves generalization and model robustness.
- Pre-trained models like VGG16 are powerful for feature extraction and fine-tuning.

## 💻 Tech Stack
- Python
- TensorFlow / Keras
- VGG16, ResNet50 (from Keras Applications)
- Jupyter Notebook

## 📂 Project Structure
├── data/
│ └── (CIFAR-10 dataset)
├── models/
│ ├── custom_cnn.py
│ ├── vgg16_model.py
│ └── resnet50_model.py
├── notebooks/
│ └── training_and_evaluation.ipynb
├── results/
│ └── evaluation_metrics.png
└── README.md


## 📈 Future Work
- Hyperparameter tuning
- Testing with newer architectures (e.g., EfficientNet, Vision Transformers)
- Expanding to more complex datasets

---
