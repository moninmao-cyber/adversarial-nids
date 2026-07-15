import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from model import NIDSModel
import joblib

# ------------------------------------------------------------------
# 1. Load and prepare data (same as before)
# ------------------------------------------------------------------
df = pd.read_csv('data/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df['is_attack'] = df[' Label'].apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
y = df['is_attack'].values

X_raw = df.select_dtypes(include=[np.number])
X_raw = X_raw.loc[:, X_raw.nunique() > 1]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw.values, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)
joblib.dump(scaler, 'models/scaler.pkl')  # re-save for consistency

# Get realistic bounds from training data (scaled space)
train_mins = X_train.min(axis=0)
train_maxs = X_train.max(axis=0)

# ------------------------------------------------------------------
# 2. Load model
# ------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_train.shape[1]
model = NIDSModel(input_dim).to(device)
model.load_state_dict(torch.load('models/baseline_nids.pth', map_location=device))
model.eval()

X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

# Clean accuracy
with torch.no_grad():
    clean_preds = torch.argmax(model(X_test_t), dim=1).cpu().numpy()
print(f"Clean accuracy: {accuracy_score(y_test, clean_preds):.4f}")

# ------------------------------------------------------------------
# 3. PGD attack function
# ------------------------------------------------------------------
def pgd_attack(model, inputs, labels, epsilon, alpha, iters, mins_t, maxs_t):
    """
    Projected Gradient Descent.
    - epsilon: maximum L∞ perturbation (how much each feature can change)
    - alpha: step size per iteration
    - iters: number of steps
    """
    # Keep a copy of the original clean input
    original = inputs.clone().detach()
    adv = inputs.clone().detach()

    for i in range(iters):
        adv.requires_grad = True
        outputs = model(adv)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        # Take a small step in the direction of the sign of the gradient
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()

            # Projection step 1: stay within epsilon-ball around the original input
            eta = torch.clamp(adv - original, -epsilon, epsilon)
            adv = original + eta

            # Projection step 2: stay within the realistic feature bounds
            adv = torch.max(adv, mins_t)
            adv = torch.min(adv, maxs_t)

    return adv

mins_t = torch.tensor(train_mins, dtype=torch.float32, device=device)
maxs_t = torch.tensor(train_maxs, dtype=torch.float32, device=device)

# ------------------------------------------------------------------
# 4. Test PGD with various epsilon values
# ------------------------------------------------------------------
# We'll fix alpha = epsilon/10 and iters=20 as a common setting.
# A smaller alpha per step with more iterations often finds better attacks.
epsilons = [0.1, 0.3, 0.5, 1.0]
print("\nPGD Attack (iters=20, alpha=epsilon/10):")
print("Epsilon | Attack Success Rate | Accuracy")
for eps in epsilons:
    alpha = eps / 10.0
    iters = 20
    adv = pgd_attack(model, X_test_t, y_test_t, eps, alpha, iters, mins_t, maxs_t)
    with torch.no_grad():
        adv_preds = torch.argmax(model(adv), dim=1).cpu().numpy()
    attack_mask = (y_test == 1)
    total_attacks = attack_mask.sum()
    missed = ((adv_preds == 0) & attack_mask).sum()
    success_rate = missed / total_attacks if total_attacks > 0 else 0
    acc = accuracy_score(y_test, adv_preds)
    print(f"  {eps:.1f}   | {success_rate:.4f}              | {acc:.4f}")

# ------------------------------------------------------------------
# 5. Optional: find minimal epsilon that achieves 95% success
# (We'll just note that for later discussion.)

