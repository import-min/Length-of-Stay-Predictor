import torch.nn as nn

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_layers = [64, 32]):
        super(NeuralNetworkModel, self).__init__()
        
        # Dynamically create hidden layers
        layers = []
        in_features = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
            
        # Output layer
        layers.append(nn.Linear(in_features, 1))
        
        # Store the layers in an nn.ModuleList
        self.layers = nn.ModuleList(layers)
            
    def forward(self, x):
        # Forward pass through the network
        for layer in self.layers:
            x = layer(x)
        return x