---
title: Interactive MLP Learning Platform
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Interactive MLP Learning Platform

This is an interactive web application designed to help students learn about Multi-Layer Perceptrons (MLPs) and deep learning concepts. The application allows users to:

1. Use data from multiple sources:
   - Generate synthetic datasets with customizable features and classes
   - Upload custom datasets (CSV format)
   - Select from standard datasets (Iris, Wine, Breast Cancer, Digits, MNIST subset)
2. Split data into training, validation, and test sets
3. Design and visualize MLP architectures (including per-layer activation functions)
4. Train MLPs and observe the learning process with real-time training and validation metrics
5. Visualize the results and model performance, including:
    - Training/validation loss and accuracy curves
    - Weight and bias visualization
    - Weight optimization over epochs
    - Network architecture diagram
    - Test image predictions (for image-based datasets)
    - Confusion matrix and classification metrics after testing

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

## Features

- Multiple dataset sources (synthetic generation, custom upload, standard datasets)
- Support for image datasets with visualization (Digits, MNIST)
- Interactive dataset generation and splitting
- Customizable MLP architecture (layers, nodes, activations)
- Real-time training and validation visualization
- Performance metrics and plots
- Weight and bias visualization
- Network architecture visualization
- Image prediction visualization for image datasets
- Confusion matrix and classification report on test data

## Usage

1. Start by selecting your dataset source:
   - Generate a random dataset with customizable parameters
   - Upload your own CSV file and select feature and target columns
   - Choose from standard datasets like Iris, Wine, Breast Cancer, etc.
2. Configure your data split (training, validation, and test percentages)
3. Design your MLP architecture (choose layers, nodes, and activations)
4. Confirm the network to visualize the architecture
5. Train the model and observe both training and validation metrics
6. Test the model on unseen data and analyze:
   - For image datasets: View sample test images with predictions
   - Confusion matrix and detailed classification metrics

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies (including: streamlit, numpy, pandas, scikit-learn, matplotlib, torch, networkx, seaborn, pillow, openml) 