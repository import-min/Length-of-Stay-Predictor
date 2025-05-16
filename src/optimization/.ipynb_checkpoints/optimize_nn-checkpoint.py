import json
import optuna
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from src.models.nn_model import NeuralNetworkModel  # Model architecture import

# Set Optuna logging level to WARNING to suppress INFO messages
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Load preprocessed training and validation sets
    X_train, y_train = torch.load(os.path.join(base_path, 'data', 'train_tensor.pt'), weights_only=True)
    X_val, y_val = torch.load(os.path.join(base_path, 'data', 'val_tensor.pt'), weights_only=True)
    
    # Change type to floats
    X_train, y_train = X_train.float(), y_train.float()
    X_val, y_val = X_val.float(), y_val.float()
    
    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    return train_loader, val_loader

def objective(trial):
    # Load data
    train_loader, val_loader = load_data()
    
    # Suggest hyperparameters using Optuna's trial object
    hidden_layers = []
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)
    
    for i in range(num_hidden_layers):
        hidden_size = trial.suggest_int(f'hidden_size_layer_{i}', 32, 128)
        hidden_layers.append(hidden_size)
        
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    
    # Initialize model with suggested hyperparameters
    input_size = next(iter(train_loader))[0].shape[1]
    model = NeuralNetworkModel(input_size=input_size, hidden_layers=hidden_layers)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # print (f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
        
        # Validation step after each epoch
        val_loss = validate_model(model, val_loader, criterion)
        
        # Report validation loss to Optuna for pruning or optimization purposes
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_loss

def validate_model(model, val_loader, criterion):
    model.eval()
    
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            running_loss += loss.item()
    
    return running_loss/len(val_loader)

def save_best_hyperparameters(study, filename='best_hyperparameters.json'):
    best_params = study.best_params
    
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f)

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    
    print('Best Trial:')
    best_trial = study.best_trial
    
    print(f'Value: {best_trial.value}')
    
    print('Best Parameters:')
    for key, value in best_trial.params.items():
        print(f'{key}: {value}')
        
    # Save best hyperparameters to file
    save_best_hyperparameters(study)