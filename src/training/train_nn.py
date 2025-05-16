import json
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from torch.utils.data import DataLoader, TensorDataset
from src.models.nn_model import NeuralNetworkModel  # Import your model architecture

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the best hyperparameters from the optimization
def load_best_hyperparameters(filename='best_hyperparameters.json'):
    with open(filename, 'r') as f:
        best_params = json.load(f)
    return best_params

# Create one DataLoader for the full dataset (train + validation)
def load_full_data():
    # Determine the base path to access data directory
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Load preprocessed train and validation sets from .pt files
    X_train, y_train = torch.load(os.path.join(base_path, 'data', 'train_tensor.pt'))
    X_val, y_val = torch.load(os.path.join(base_path, 'data', 'val_tensor.pt'))
    
    # Concatenate train and validation datasets
    X_full = torch.cat((X_train, X_val), dim=0)
    y_full = torch.cat((y_train, y_val), dim=0)
    
    # Ensure tensors are of type float
    X_full = X_full.float()
    y_full = y_full.float()
    
    # Create DataLoader for batching
    full_dataset = TensorDataset(X_full, y_full)
    full_loader = DataLoader(full_dataset, batch_size=32)
    
    return full_loader

# Initialize the model with the best hyperparameters
def initialize_model(best_params):
    # Extract hyperparameters from JSON file
    hidden_layers = [best_params[f'hidden_size_layer_{i}'] for i in range(best_params['num_hidden_layers'])]
    
    learning_rate = best_params['learning_rate']
    
    # Determine input size from dataset by loading the .pt file
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    X_train, _ = torch.load(os.path.join(base_path, 'data', 'train_tensor.pt'))
    
    input_size = X_train.shape[1]  # Number of features in dataset
    
    # Initialize model with optimal hyperparameters
    model = NeuralNetworkModel(input_size=input_size, hidden_layers=hidden_layers)
    
    # Initialize optimizer with optimal learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, optimizer

# Train final model
def train_final_model(model, optimizer, full_loader):
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks

    epochs = 20

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in full_loader:
            optimizer.zero_grad()  # Reset gradients
            
            outputs = model(inputs)  # Forward pass through the model
            
            loss = criterion(outputs.squeeze(), targets)  # Calculate loss
            
            loss.backward()  # Backpropagation
            
            optimizer.step()  # Update weights
            
            running_loss += loss.item()
        
        # print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(full_loader)}')

def save_model(model, filename='nn_model.pth'):
    torch.save(model.state_dict(), f'final-models/{filename}')

if __name__ == '__main__':
    best_params = load_best_hyperparameters()
    full_loader = load_full_data()
    model, optimizer = initialize_model(best_params)
    train_final_model(model, optimizer, full_loader)
    torch.save(model.state_dict(), 'final-models/nn_model.pth')