import torch
import torch.nn as nn
import os

class DummyModel(nn.Module):
    def __init__(self, input_size):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
    def forward(self, x):
        return self.fc(x)

def load_tensors():
    base_path = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(base_path, 'data', 'train_tensor.pt')
    train_data = torch.load(train_path, weights_only=True)
    features, targets = train_data
    return features, targets

def main():
    features, targets = load_tensors()
    input_size = features.shape[1]
    model = DummyModel(input_size)
    with torch.no_grad():
        output = model(features)
    print("Output sample:", output[0].item())
    
if __name__ == "__main__":
    main()
