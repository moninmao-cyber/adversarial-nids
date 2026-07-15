import torch
import torch.nn as nn

class NIDSModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 2)   # binary: benign or attack

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
