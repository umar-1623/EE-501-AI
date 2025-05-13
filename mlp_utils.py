import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import networkx as nx
from sklearn import datasets

# Supported activations
ACTIVATION_MAP = {
    'ReLU': nn.ReLU(),
    'Tanh': nn.Tanh(),
    'Sigmoid': nn.Sigmoid(),
    'LeakyReLU': nn.LeakyReLU(),
    'Identity': nn.Identity()
}

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activations):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = []
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.activations.append(ACTIVATION_MAP[activations[0]])
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.activations.append(ACTIVATION_MAP[activations[i+1]])
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.activations.append(ACTIVATION_MAP['Identity'])  # No activation for output
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activations[i](layer(x))
        x = self.layers[-1](x)
        return self.softmax(x)

def generate_dataset(n_samples, n_features, n_classes, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features,
        n_redundant=0,
        random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def load_custom_dataset(df, feature_columns, target_column):
    """
    Load and preprocess a custom dataset from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the dataset
        feature_columns (list): List of column names to use as features
        target_column (str): Name of the column to use as target
    
    Returns:
        tuple: (X, y, n_features, n_classes)
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            n_features (int): Number of features
            n_classes (int): Number of classes
    """
    # Extract features and target
    X = df[feature_columns].values
    y = df[target_column].values
    
    # Handle categorical targets
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Get dimensions
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    return X, y, n_features, n_classes

def load_standard_dataset(dataset_name):
    """
    Load a standard dataset from scikit-learn.
    
    Args:
        dataset_name (str): Name of the dataset to load
    
    Returns:
        tuple: (X, y, n_features, n_classes)
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            n_features (int): Number of features
            n_classes (int): Number of classes
    """
    is_image_data = False
    img_shape = None
    
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Digits":
        data = datasets.load_digits()
        is_image_data = True
        img_shape = (8, 8)  # 8x8 pixel images
    elif dataset_name == "MNIST (subset)":
        # Load a subset of MNIST
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
        X = X.to_numpy()
        y = y.astype(int).to_numpy()
        
        # Take a subset of 2000 samples
        indices = np.random.RandomState(42).choice(len(X), 2000, replace=False)
        X = X[indices]
        y = y[indices]
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        is_image_data = True
        img_shape = (28, 28)  # 28x28 pixel images
        
        return X, y, X.shape[1], len(np.unique(y)), is_image_data, img_shape
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X = data.data
    y = data.target
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    return X, y, n_features, n_classes, is_image_data, img_shape

def split_data(X, y, val_pct, test_pct, random_state=42):
    np.random.seed(random_state)
    n = X.shape[0]
    idx = np.random.permutation(n)
    n_test = int(n * test_pct)
    n_val = int(n * val_pct)
    n_train = n - n_val - n_test
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])

def train_model(model, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size=32, track_weights=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    n_samples = X_train.shape[0]
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    weights_history = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        indices = torch.randperm(n_samples)
        X_shuffled = X_train_tensor[indices]
        y_shuffled = y_train_tensor[indices]
        
        epoch_train_loss = 0
        train_correct = 0
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == batch_y).sum().item()
        
        # Calculate average training loss and accuracy
        avg_train_loss = epoch_train_loss / (n_samples / batch_size)
        train_accuracy = train_correct / n_samples
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_correct = (val_predicted == y_val_tensor).sum().item()
            val_accuracy = val_correct / len(y_val)
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy)
        
        if track_weights:
            weights_history.append(model.layers[0].weight.detach().cpu().numpy().copy())
    
    return (train_losses, train_accuracies, val_losses, val_accuracies, weights_history) if track_weights else (train_losses, train_accuracies, val_losses, val_accuracies)

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, n_classes):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(n_classes)],
                yticklabels=[f'Class {i}' for i in range(n_classes)])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    return plt.gcf()

def plot_classification_metrics(y_true, y_pred, n_classes):
    from sklearn.metrics import classification_report
    import pandas as pd
    
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df = df.drop('support', axis=1)
    df = df.round(3)
    
    return df

def visualize_weights(model):
    weights = []
    for layer in model.layers:
        weights.append(layer.weight.detach().numpy())
    
    n_layers = len(weights)
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 5))
    if n_layers == 1:
        axes = [axes]
    for i, (weight, ax) in enumerate(zip(weights, axes)):
        im = ax.imshow(weight, cmap='coolwarm')
        ax.set_title(f'Layer {i+1} Weights')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig

def plot_weight_optimization(weights_history):
    # Visualize the change of the first weight in the first neuron over epochs
    weights_history = np.array(weights_history)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(weights_history.shape[1]):
        ax.plot(weights_history[:, i, 0], label=f'Neuron {i+1}')
    ax.set_title('First Layer Weights Optimization (first input weight per neuron)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Value')
    ax.legend()
    plt.tight_layout()
    return fig

def visualize_network(input_size, hidden_sizes, output_size):
    G = nx.DiGraph()
    layers = [input_size] + hidden_sizes + [output_size]
    pos = {}
    node_labels = {}
    node_count = 0
    y_gap = 1.5
    x_gap = 2
    for l, n_nodes in enumerate(layers):
        for n in range(n_nodes):
            node_id = f'L{l}N{n}'
            G.add_node(node_id, layer=l)
            pos[node_id] = (l * x_gap, -n * y_gap + (n_nodes-1)*y_gap/2)
            if l == 0:
                node_labels[node_id] = f'In{n+1}'
            elif l == len(layers)-1:
                node_labels[node_id] = f'Out{n+1}'
            else:
                node_labels[node_id] = f'H{l}-{n+1}'
    # Add edges
    for l in range(len(layers)-1):
        for n1 in range(layers[l]):
            for n2 in range(layers[l+1]):
                G.add_edge(f'L{l}N{n1}', f'L{l+1}N{n2}')
    fig, ax = plt.subplots(figsize=(2*len(layers), 6))
    nx.draw(G, pos, ax=ax, with_labels=True, labels=node_labels, node_size=1000, node_color='skyblue', arrowsize=10)
    ax.set_title('MLP Architecture')
    plt.tight_layout()
    return fig

def plot_image_predictions(X_test, y_test, y_pred, img_shape, num_samples=10):
    """
    Plot test images with their true and predicted labels.
    
    Args:
        X_test (np.ndarray): Test images (flattened)
        y_test (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        img_shape (tuple): Shape of original images (height, width)
        num_samples (int): Number of samples to display
    
    Returns:
        matplotlib.figure.Figure: Figure with plotted images
    """
    # Select a subset of images to display
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        # Reshape and normalize the image for display
        img = X_test[idx].reshape(img_shape)
        
        # Convert standardized values back to 0-1 range for display
        img_min, img_max = img.min(), img.max()
        img_normalized = (img - img_min) / (img_max - img_min)
        
        # Display the image
        axes[i].imshow(img_normalized, cmap='gray')
        axes[i].set_title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}')
        axes[i].axis('off')
        
        # Add color to the title based on correct/incorrect prediction
        if y_test[idx] == y_pred[idx]:
            axes[i].title.set_color('green')
        else:
            axes[i].title.set_color('red')
    
    plt.tight_layout()
    return fig 