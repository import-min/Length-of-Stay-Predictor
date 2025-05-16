import os
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, r2_score
from src.models.nn_model import NeuralNetworkModel
from src.training.train_nn import load_best_hyperparameters
from torch.utils.data import DataLoader, TensorDataset

def load_test_data():
    # Determine the base path to access data directory
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Load test set from .pt file
    X_test, y_test = torch.load(os.path.join(base_path, 'data', 'test_tensor.pt'))
    
    # Ensure tensors are of type float
    X_test = X_test.float()
    y_test = y_test.float()
    
    # Create DataLoader for batching
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return test_loader

def load_trained_model():
    best_params = load_best_hyperparameters()
    
    hidden_layers = [best_params[f'hidden_size_layer_{i}'] for i in range(best_params['num_hidden_layers'])]
    
    # Determine input size from dataset by loading the .pt file
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    X_train, _ = torch.load(os.path.join(base_path, 'data', 'train_tensor.pt'))
    
    input_size = X_train.shape[1]  # Number of features in dataset
    
    # Initialize model with same architecture as during training (use best hyperparameters)
    model = NeuralNetworkModel(input_size=input_size, hidden_layers=hidden_layers)
    
    # Load saved weights from training
    model.load_state_dict(torch.load('final-models/nn_model.pth'))
    
    return model

def evaluate_model(model, test_loader):
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
            
            all_predictions.extend(outputs.numpy())
            all_targets.extend(targets.numpy())
            
    avg_loss = total_loss / len(test_loader)
    
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    print(f'Test MSE: {avg_loss}')
    print(f'Test MAE: {mae}')
    print(f'Test R-squared: {r2}')
    
if __name__ == '__main__':
    test_loader = load_test_data()
    model = load_trained_model()
    
    evaluate_model(model, test_loader)