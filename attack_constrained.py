import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from model import NIDSModel

# ------------------------------------------------------------------
# 1. Load raw data (same CSV, same cleaning as always)
# ------------------------------------------------------------------
df = pd.read_csv('data/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df['is_attack'] = df[' Label'].apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
y = df['is_attack'].values

# Keep only numeric columns that have more than one unique value
X_raw = df.select_dtypes(include=[np.number])
X_raw = X_raw.loc[:, X_raw.nunique() > 1]   # drop useless columns

# ------------------------------------------------------------------
# 2. Train/test split (exactly the same random_state as before)
# ------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw.values, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------------
# 3. Scale the data and save the scaler
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# Save the scaler for later use (so we never mismatch again)
import joblib
joblib.dump(scaler, 'models/scaler.pkl')

# ------------------------------------------------------------------
# 4. Compute feasible range for each feature (in scaled space)
#    We use the training set's min/max to define "realistic" values.
# ------------------------------------------------------------------
train_mins = X_train.min(axis=0)   # per‑feature minimum in scaled space
train_maxs = X_train.max(axis=0)   # per‑feature maximum in scaled space

# For extra safety, we can slightly widen the range (0.1% margin)
# but let's keep it exact for now.

# ------------------------------------------------------------------
# 5. Load the trained model
# ------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_train.shape[1]
model = NIDSModel(input_dim).to(device)
model.load_state_dict(torch.load('models/baseline_nids.pth', map_location=device))
model.eval()

# Convert test data to tensors
X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

# Clean accuracy
with torch.no_grad():
    clean_preds = torch.argmax(model(X_test_t), dim=1).cpu().numpy()
clean_acc = accuracy_score(y_test, clean_preds)
print(f"Clean accuracy: {clean_acc:.4f}")

# ------------------------------------------------------------------
# 6. Constrained FGSM attack
# ------------------------------------------------------------------
def constrained_fgsm(model, inputs, labels, epsilon, mins_tensor, maxs_tensor):
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * inputs.grad.sign()
    adv = inputs + perturbation
    # Clip each feature to the allowed range (the realistic min/max from training data)
    adv = torch.max(adv, mins_tensor)
    adv = torch.min(adv, maxs_tensor)
    return adv

mins_t = torch.tensor(train_mins, dtype=torch.float32, device=device)
maxs_t = torch.tensor(train_maxs, dtype=torch.float32, device=device)

epsilons = [0.1, 0.5, 1.0, 2.0]
print("\nConstrained FGSM (features kept within training min–max):")
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
