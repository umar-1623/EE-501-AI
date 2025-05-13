import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import io
from sklearn import datasets
from mlp_utils import (
    MLP, generate_dataset, split_data, train_model, plot_training_history,
    visualize_weights, plot_weight_optimization, visualize_network, 
    plot_confusion_matrix, plot_classification_metrics, ACTIVATION_MAP,
    load_custom_dataset, load_standard_dataset, plot_image_predictions
)

st.set_page_config(page_title="Interactive MLP Learning Platform", layout="wide")

st.title("Interactive MLP Learning Platform")
st.markdown("""
This application helps you learn about Multi-Layer Perceptrons (MLPs) through interactive experimentation.
You can generate synthetic data, upload your own dataset, select standard datasets, design your own MLP architecture, and observe the training process.
""")

# Sidebar for dataset configuration
st.sidebar.header("Dataset Configuration")

# Dataset source selection
dataset_source = st.sidebar.radio(
    "Dataset Source",
    ["Generate Random Dataset", "Upload Custom Dataset", "Use Standard Dataset"]
)

if dataset_source == "Generate Random Dataset":
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500)
    n_features = st.sidebar.slider("Number of Features", 2, 10, 4)
    n_classes = st.sidebar.slider("Number of Classes", 2, 5, 3)
    
    # Data split percentages
    st.sidebar.subheader("Data Split (%)")
    def_percent = 20
    val_percent = st.sidebar.slider("Validation %", 0, 50, def_percent)
    test_percent = st.sidebar.slider("Test %", 0, 50, def_percent)
    train_percent = 100 - val_percent - test_percent
    if train_percent < 1:
        st.sidebar.error("Train % must be at least 1%.")

    # Generate dataset
    if st.sidebar.button("Generate Dataset"):
        X, y = generate_dataset(n_samples, n_features, n_classes)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
            X, y, val_percent/100, test_percent/100)
        st.session_state['X_train'] = X_train
        st.session_state['y_train'] = y_train
        st.session_state['X_val'] = X_val
        st.session_state['y_val'] = y_val
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['n_features'] = n_features
        st.session_state['n_classes'] = n_classes
        st.session_state['is_image_data'] = False
        st.session_state['dataset_generated'] = True
        st.session_state['network_confirmed'] = False
        st.session_state['training_complete'] = False
        st.session_state['testing_complete'] = False

elif dataset_source == "Upload Custom Dataset":
    st.sidebar.subheader("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.sidebar.info("File uploaded successfully!")
        
        # Column selection
        st.sidebar.subheader("Column Configuration")
        df = pd.read_csv(uploaded_file)
        all_columns = df.columns.tolist()
        
        # Select target column
        target_column = st.sidebar.selectbox("Select Target Column", all_columns)
        
        # Select feature columns
        feature_columns = st.sidebar.multiselect(
            "Select Feature Columns", 
            [col for col in all_columns if col != target_column],
            default=[col for col in all_columns if col != target_column]
        )
        
        # Data split percentages
        st.sidebar.subheader("Data Split (%)")
        def_percent = 20
        val_percent = st.sidebar.slider("Validation %", 0, 50, def_percent)
        test_percent = st.sidebar.slider("Test %", 0, 50, def_percent)
        train_percent = 100 - val_percent - test_percent
        if train_percent < 1:
            st.sidebar.error("Train % must be at least 1%.")
        
        # Process dataset
        if st.sidebar.button("Process Dataset"):
            X, y, n_features, n_classes = load_custom_dataset(df, feature_columns, target_column)
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
                X, y, val_percent/100, test_percent/100)
                
            st.session_state['X_train'] = X_train
            st.session_state['y_train'] = y_train
            st.session_state['X_val'] = X_val
            st.session_state['y_val'] = y_val
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['n_features'] = n_features
            st.session_state['n_classes'] = n_classes
            st.session_state['is_image_data'] = False
            st.session_state['dataset_generated'] = True
            st.session_state['network_confirmed'] = False
            st.session_state['training_complete'] = False
            st.session_state['testing_complete'] = False

elif dataset_source == "Use Standard Dataset":
    st.sidebar.subheader("Standard Dataset Selection")
    dataset_options = [
        "Iris",
        "Wine",
        "Breast Cancer",
        "Digits",
        "MNIST (subset)"
    ]
    selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_options)
    
    # Provide dataset info
    dataset_info = {
        "Iris": "150 samples, 4 features, 3 classes - Iris flower species",
        "Wine": "178 samples, 13 features, 3 classes - Wine cultivars",
        "Breast Cancer": "569 samples, 30 features, 2 classes - Malignant/Benign tumors",
        "Digits": "1797 samples, 64 features, 10 classes - Handwritten digits (8x8 pixels)",
        "MNIST (subset)": "2000 samples, 784 features, 10 classes - Handwritten digits (28x28 pixels)"
    }
    
    st.sidebar.info(dataset_info[selected_dataset])
    
    # Data split percentages
    st.sidebar.subheader("Data Split (%)")
    def_percent = 20
    val_percent = st.sidebar.slider("Validation %", 0, 50, def_percent)
    test_percent = st.sidebar.slider("Test %", 0, 50, def_percent)
    train_percent = 100 - val_percent - test_percent
    if train_percent < 1:
        st.sidebar.error("Train % must be at least 1%.")
    
    # Load standard dataset
    if st.sidebar.button("Load Dataset"):
        X, y, n_features, n_classes, is_image_data, img_shape = load_standard_dataset(selected_dataset)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
            X, y, val_percent/100, test_percent/100)
            
        st.session_state['X_train'] = X_train
        st.session_state['y_train'] = y_train
        st.session_state['X_val'] = X_val
        st.session_state['y_val'] = y_val
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['n_features'] = n_features
        st.session_state['n_classes'] = n_classes
        st.session_state['is_image_data'] = is_image_data
        if is_image_data:
            st.session_state['img_shape'] = img_shape
        st.session_state['dataset_generated'] = True
        st.session_state['network_confirmed'] = False
        st.session_state['training_complete'] = False
        st.session_state['testing_complete'] = False

# Main content area
if 'dataset_generated' in st.session_state:
    st.header("Dataset Information")
    st.write(f"Train: {len(st.session_state['X_train'])} samples | "
             f"Validation: {len(st.session_state['X_val'])} samples | "
             f"Test: {len(st.session_state['X_test'])} samples")
    
    # Display dataset statistics and preview
    n_features = st.session_state['n_features']
    n_classes = st.session_state['n_classes']
    
    # If image data, show sample images
    if st.session_state.get('is_image_data', False):
        st.subheader("Sample Images")
        img_shape = st.session_state['img_shape']
        
        # Display a few sample images from the training set
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, idx in enumerate(np.random.choice(len(st.session_state['X_train']), 5)):
            img = st.session_state['X_train'][idx].reshape(img_shape)
            img_min, img_max = img.min(), img.max()
            img_normalized = (img - img_min) / (img_max - img_min)
            axes[i].imshow(img_normalized, cmap='gray')
            axes[i].set_title(f"Class: {st.session_state['y_train'][idx]}")
            axes[i].axis('off')
        st.pyplot(fig)
    
    # Display table with feature values
    df = pd.DataFrame(st.session_state['X_train'], columns=[f'Feature {i+1}' for i in range(n_features)])
    df['Class'] = st.session_state['y_train']
    st.subheader("Training Set Preview")
    st.dataframe(df.head())
    
    # MLP Configuration
    st.header("MLP Configuration")
    n_hidden_layers = st.slider("Number of Hidden Layers", 1, 5, 2)
    hidden_sizes = []
    activations = []
    activation_options = list(ACTIVATION_MAP.keys())
    for i in range(n_hidden_layers):
        cols = st.columns([2, 2])
        with cols[0]:
            size = st.slider(f"Nodes in Hidden Layer {i+1}", 2, 20, 8, key=f"hsize_{i}")
        with cols[1]:
            act = st.selectbox(f"Activation for Layer {i+1}", activation_options[:-1], index=0, key=f"act_{i}")
        hidden_sizes.append(size)
        activations.append(act)
    # Add activation for input to first hidden
    activations = [activations[0]] + activations
    
    # Confirm network button
    if st.button("Confirm Network"):
        st.session_state['hidden_sizes'] = hidden_sizes
        st.session_state['activations'] = activations
        st.session_state['network_confirmed'] = True
        st.session_state['training_complete'] = False
        st.session_state['testing_complete'] = False
    
    # Show network configuration
    if st.session_state.get('network_confirmed', False):
        st.subheader("Network Architecture Visualization")
        fig = visualize_network(n_features, hidden_sizes, n_classes)
        st.pyplot(fig)
        st.write(f"Input: {n_features} | Hidden: {hidden_sizes} | Output: {n_classes}")
        st.write(f"Activations: {st.session_state['activations']}")
        
        # Training parameters
        st.subheader("Training Parameters")
        epochs = st.slider("Number of Epochs", 10, 200, 50)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        batch_size = st.slider("Batch Size", 8, 128, 32)
        
        # Train button
        if st.button("Train MLP"):
            model = MLP(n_features, hidden_sizes, n_classes, st.session_state['activations'])
            train_losses, train_accuracies, val_losses, val_accuracies, weights_history = train_model(
                model,
                st.session_state['X_train'],
                st.session_state['y_train'],
                st.session_state['X_val'],
                st.session_state['y_val'],
                epochs,
                learning_rate,
                batch_size,
                track_weights=True
            )
            st.session_state['model'] = model
            st.session_state['train_losses'] = train_losses
            st.session_state['train_accuracies'] = train_accuracies
            st.session_state['val_losses'] = val_losses
            st.session_state['val_accuracies'] = val_accuracies
            st.session_state['weights_history'] = weights_history
            st.session_state['training_complete'] = True
            st.session_state['testing_complete'] = False
        
        # Show training results
        if st.session_state.get('training_complete', False):
            st.header("Training Results")
            fig = plot_training_history(
                st.session_state['train_losses'],
                st.session_state['train_accuracies'],
                st.session_state['val_losses'],
                st.session_state['val_accuracies']
            )
            st.pyplot(fig)
            
            st.subheader("Weight Visualization (All Layers)")
            weight_fig = visualize_weights(st.session_state['model'])
            st.pyplot(weight_fig)
            
            st.subheader("Weight Optimization (First Layer)")
            opt_fig = plot_weight_optimization(st.session_state['weights_history'])
            st.pyplot(opt_fig)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Training Loss", f"{st.session_state['train_losses'][-1]:.4f}")
            with col2:
                st.metric("Final Training Accuracy", f"{st.session_state['train_accuracies'][-1]:.2%}")
            with col3:
                st.metric("Final Validation Loss", f"{st.session_state['val_losses'][-1]:.4f}")
            with col4:
                st.metric("Final Validation Accuracy", f"{st.session_state['val_accuracies'][-1]:.2%}")
            
            # Test button
            if st.button("Test on Unseen Data"):
                model = st.session_state['model']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test)
                    outputs = model(X_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    test_accuracy = (predicted.numpy() == y_test).mean()
                
                st.session_state['test_accuracy'] = test_accuracy
                st.session_state['test_predictions'] = predicted.numpy()
                st.session_state['testing_complete'] = True
            
            if st.session_state.get('testing_complete', False):
                st.header("Test Results")
                st.success(f"Test Accuracy: {st.session_state['test_accuracy']:.2%}")
                
                # Display test images for image data
                if st.session_state.get('is_image_data', False):
                    st.subheader("Test Image Predictions")
                    img_fig = plot_image_predictions(
                        st.session_state['X_test'],
                        st.session_state['y_test'],
                        st.session_state['test_predictions'],
                        st.session_state['img_shape']
                    )
                    st.pyplot(img_fig)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm_fig = plot_confusion_matrix(
                    st.session_state['y_test'],
                    st.session_state['test_predictions'],
                    n_classes
                )
                st.pyplot(cm_fig)
                
                # Classification Metrics
                st.subheader("Classification Metrics")
                metrics_df = plot_classification_metrics(
                    st.session_state['y_test'],
                    st.session_state['test_predictions'],
                    n_classes
                )
                st.dataframe(metrics_df)
                
                # Additional Test Metrics
                st.subheader("Additional Test Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Test Accuracy", f"{st.session_state['test_accuracy']:.2%}")
                with col2:
                    st.metric("Test Error Rate", f"{1 - st.session_state['test_accuracy']:.2%}")
else:
    st.info("Please select a dataset source from the sidebar to begin.") 