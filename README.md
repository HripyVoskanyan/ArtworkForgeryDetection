# ArtworkFraudDetection

This repository is part of a Master's thesis project aimed at detecting AI-generated forgeries in artwork. By leveraging a dataset poisoned with adversarially generated fake images, the project explores the potential of machine learning models to identify and differentiate between genuine and fake artwork.  

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Setup](#setup)
- [Future Work](#future-work)

---

## Overview
The **ArtworkFraudDetection** project is an experimental study to improve art fraud detection systems. It combines adversarial attack techniques, neural networks, and various machine learning models to classify artworks as genuine or fake. 

This project introduces a novel approach by poisoning the dataset using multiple faking techniques and benchmarking their effectiveness in detection tasks.

---

## Features
- Creation of fake artwork using diverse techniques 
- Integration of multiple dataset augmentation and poisoning methods.
- Classification of artworks using pretrained models such as ResNet and Vision Transformers (ViTs).
- Exploration of self-supervised learning methods like DINO embeddings for enhanced classification.
- Comprehensive data pipeline for preprocessing, training, and evaluation.

---

## Dataset
- **Original Dataset:** [WikiArt](https://www.wikiart.org/)
  - Contains ~80,000 original artwork images.
- **Poisoned Dataset:** AI-generated fake images created using the following techniques:
  - Neural style transfer
  - Gaussian noise and traditional editing
  - Adversarial attack methods (FGSM)
  - Generative AI
- **Dataset Structure:**  
  - `data/original`: Contains genuine artwork. Please install the dataset in that folder, or use deeplake's version of it.
  - `data/poisoned`: Contains fake artwork generated using different methods.  

All images are resized to a uniform size for compatibility with the classification models.

---

## Methodology
### 1. Data Preparation:
   - Merge the original and poisoned datasets.
   - Resize images and apply augmentation.
   - Split the dataset into training, validation, and testing sets.

### 2. Poisoning:
   - Use the `UniversalFaker` class to apply various poisoning techniques.

### 3. Classification:
   - Use pretrained models (ResNet, ViT) to classify images.
   - Experiment with self-supervised learning models like DINO for improved feature extraction.

### 4. Evaluation:
   - Compare model performance across poisoning methods.
   - Evaluate metrics: accuracy, precision, recall, F1-score.

---

## Setup
### Prerequisites
- Python 3.9+
- [PyTorch](https://pytorch.org/get-started/locally/)
- NVIDIA GPU for training (recommended)
- Required packages: `torchvision`, `opencv-python`, `scikit-learn`, `matplotlib`, `deeplake`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/HripyVoskanyan/ArtworkForgeryDetection.git
   cd ArtworkForgeryDetection
   ```
2. Set up a virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  ```
3. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
4. Dataset Preparation
  ```bash
  python scripts/run_poisoning.py
  ```
5. Train the Methods
  ```bash
  python classification/main.py
  ```

---

## Future Work
- Extend poisoning techniques to include newer generative models.
- Fine-tune self-supervised learning models for higher classification accuracy.
- Explore additional augmentation strategies for enhanced robustness.
