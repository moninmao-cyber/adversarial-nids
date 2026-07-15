import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from model import NIDSModel

# Load raw data and constraints
raw_data = np.load('data/ids2017_ddos_test.npz', allow_pickle=True)  # but we need raw, not scaled
# Actually, we saved only scaled data. We'll re-read the CSV for raw test set later.
# For simplicity, let's load the raw CSV again and split exactly like preprocess.
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df['is_attack'] = df[' Label'].apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
y = df['is_attack'].values
X_raw = df.select_dtypes(include=[np.number])
X_raw = X_raw.loc[:, X_raw.nunique() > 1]  # matches earlier drop

# Split exactly the same way (random_state=42, stratify)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw.values, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data: fit scaler on training raw data, then transform both
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Load feature constraints from raw data (mins, maxs)
constraints = np.load('data/feature_constraints.npz')
mins = constraints['mins']
maxs = constraints['maxs']

# Transform those min/max into scaled space
mins_scaled = scaler.transform(mins.reshape(1, -1)).flatten()
maxs_scaled = scaler.transform(maxs.reshape(1, -1)).flatten()

# Now convert test data to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_test.shape[1]
model = NIDSModel(input_dim).to(device)
model.load_state_dict(torch.load('models/baseline_nids.pth', map_location=device))
model.eval()

X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

# Clean accuracy
with torch.no_grad():
    clean_preds = torch.argmax(model(X_test_t), dim=1).cpu().numpy()
clean_acc = accuracy_score(y_test, clean_preds)
print(f"Clean accuracy: {clean_acc:.4f}")

def constrained_fgsm(model, inputs, labels, epsilon, mins_tensor, maxs_tensor):
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * inputs.grad.sign()
    adv = inputs + perturbation
    # Project onto feasible range per feature
    adv = torch.max(adv, mins_tensor)
    adv = torch.min(adv, maxs_tensor)
    return adv

mins_t = torch.tensor(mins_scaled, dtype=torch.float32, device=device)
maxs_t = torch.tensor(maxs_scaled, dtype=torch.float32, device=device)

epsilons = [0.1, 0.5, 1.0, 2.0]
print("\nConstrained FGSM Results (within original feature ranges):")
print("Epsilon | Attack Success Rate | Accuracy")
for eps in epsilons:
    adv = constrained_fgsm(model, X_test_t, y_test_t, eps, mins_t, maxs_t)
    with torch.no_grad():
        adv_preds = torch.argmax(model(adv), dim=1).cpu().numpy()
    attack_mask = (y_test == 1)
    total_attacks = attack_mask.sum()
    missed = ((adv_preds == 0) & attack_mask).sum()
    success_rate = missed / total_attacks if total_attacks > 0 else 0
    acc = accuracy_score(y_test, adv_preds)
    print(f"  {eps:.1f}   | {success_rate:.4f}              | {acc:.4f}")
