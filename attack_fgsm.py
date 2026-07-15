import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from model import NIDSModel

# ------------------------------------------------------------
# What is FGSM?
# FGSM = Fast Gradient Sign Method.
# We compute the gradient of the loss with respect to the input,
# take its sign (+1 or -1), multiply by a tiny number epsilon,
# and add that to the original input. The model often misclassifies it.
# ------------------------------------------------------------

def fgsm_attack(model, inputs, labels, epsilon):
    """
    Generates adversarial examples for a batch of inputs.
    model: trained NIDS model
    inputs: batch of network flow features (tensor)
    labels: true labels (tensor) - we use these to compute the loss direction
    epsilon: maximum perturbation per feature (small, e.g., 0.01)
    """
    # Tell PyTorch we need gradients for the inputs
    inputs.requires_grad = True

    # Forward pass
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    # Compute gradient of loss with respect to the input features
    model.zero_grad()
    loss.backward()

    # The gradient tells us the direction that would INCREASE the loss the most.
    # We take its sign and scale by epsilon.
    perturbation = epsilon * inputs.grad.sign()

    # Create adversarial example by adding perturbation
    adv_inputs = inputs + perturbation

    # Optional: clip to a reasonable range (since our data is standardized,
    # most values lie between -3 and 3. We clip to [-10, 10] to avoid extreme values.)
    adv_inputs = torch.clamp(adv_inputs, -10.0, 10.0)

    return adv_inputs


# ------------------------------------------------------------
# 1. Load the preprocessed data and the trained model
# ------------------------------------------------------------
data = np.load('data/ids2017_ddos_test.npz')
X_test = data['X_test']
y_test = data['y_test']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert test data to tensor (we only need test set for evaluation)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# Load the saved model
input_dim = X_test.shape[1]
model = NIDSModel(input_dim).to(device)
model.load_state_dict(torch.load('models/baseline_nids.pth', map_location=device))
model.eval()  # evaluation mode (no dropout, etc.)

print("Model loaded. Ready to attack.")


# ------------------------------------------------------------
# 2. Evaluate clean accuracy first (baseline)
# ------------------------------------------------------------
with torch.no_grad():
    clean_outputs = model(X_test_t)
    clean_preds = torch.argmax(clean_outputs, dim=1).cpu().numpy()
clean_acc = accuracy_score(y_test, clean_preds)
print(f"Clean test accuracy: {clean_acc:.4f}")


# ------------------------------------------------------------
# 3. Run FGSM attack with different epsilon values
# ------------------------------------------------------------
epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]

print("\nFGSM Attack Results:")
print("Epsilon | Attack success rate (how many true attacks became 'benign')")
print("--------|------------------------------------------------------------")

for eps in epsilons:
    # Generate adversarial examples for the whole test set
    adv_inputs = fgsm_attack(model, X_test_t, y_test_t, eps)

    # Predict on adversarial examples
    with torch.no_grad():
        adv_outputs = model(adv_inputs)
        adv_preds = torch.argmax(adv_outputs, dim=1).cpu().numpy()

    # Attack success rate: among samples that were originally attacks (label=1),
    # what fraction is now classified as benign (0)?
    attack_mask = (y_test == 1)  # true attacks
    orig_attack_total = attack_mask.sum()
    # How many of those true attacks are now predicted as benign?
    missed_attacks = ((adv_preds == 0) & attack_mask).sum()
    success_rate = missed_attacks / orig_attack_total if orig_attack_total > 0 else 0

    # Overall accuracy
    adv_acc = accuracy_score(y_test, adv_preds)

    print(f" {eps:.3f}   | {success_rate:.4f}  (accuracy dropped to {adv_acc:.4f})")
