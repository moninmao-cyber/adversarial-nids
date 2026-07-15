import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from model import NIDSModel

# Load data and model
data = np.load('data/ids2017_ddos_test.npz')
X_test = data['X_test'][:1000]  # just 1000 samples for speed
y_test = data['y_test'][:1000]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_test.shape[1]
model = NIDSModel(input_dim).to(device)
model.load_state_dict(torch.load('models/baseline_nids.pth', map_location=device))
model.eval()

X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
y_t = torch.tensor(y_test, dtype=torch.long, device=device)

# Check clean accuracy
with torch.no_grad():
    clean_out = model(X_t)
    clean_preds = torch.argmax(clean_out, dim=1).cpu().numpy()
print(f"Clean accuracy on 1000 samples: {accuracy_score(y_test, clean_preds):.4f}")

# FGSM with gradient debug
def debug_fgsm(model, inputs, labels, epsilon):
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad = inputs.grad.data
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient L2 norm: {grad.norm().item():.6f}")
    print(f"  Gradient mean abs: {grad.abs().mean().item():.6f}")
    print(f"  Fraction of non-zero gradient signs: {(grad.sign() != 0).float().mean().item():.4f}")
    pert = epsilon * grad.sign()
    print(f"  Perturbation L2 norm: {pert.norm().item():.6f}")
    adv = inputs + pert
    adv = torch.clamp(adv, -10, 10)
    return adv

# Try larger epsilons
for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
    print(f"\nEpsilon: {eps}")
    adv = debug_fgsm(model, X_t, y_t, eps)
    with torch.no_grad():
        adv_out = model(adv)
        adv_preds = torch.argmax(adv_out, dim=1).cpu().numpy()
    # success rate among true attacks
    attack_mask = (y_test == 1)
    total_attacks = attack_mask.sum()
    missed = ((adv_preds == 0) & attack_mask).sum()
    adv_acc = accuracy_score(y_test, adv_preds)
    print(f"  Attack success rate: {missed/total_attacks:.4f}, overall accuracy: {adv_acc:.4f}")
