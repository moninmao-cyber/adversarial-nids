import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from model import NIDSModel

# Load full data
data = np.load('data/ids2017_full_balanced.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t  = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t  = torch.tensor(y_test, dtype=torch.long).to(device)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)  # larger batch
test_loader  = DataLoader(test_ds, batch_size=2048, shuffle=False)

input_dim = X_train.shape[1]
model = NIDSModel(input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train more epochs because data is larger
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss/len(train_loader):.4f}")

# Evaluate
model.eval()
all_preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        outputs = model(batch_x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())

print("\nClassification Report (Full Dataset):")
print(classification_report(y_test, all_preds, target_names=['BENIGN', 'ATTACK']))

# Save model
torch.save(model.state_dict(), 'models/baseline_nids_full.pth')
print("Model saved to models/baseline_nids_full.pth")
