
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import joblib
from model import NIDSModel
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))  # to import from src/

# ------------------------------------------------------------------
# 1. Load the full balanced dataset and model
# ------------------------------------------------------------------
data = np.load('data/ids2017_full_balanced.npz')
X_test = data['X_test']
y_test = data['y_test']

scaler = joblib.load('models/scaler_full.pkl')
# We need the feature bounds from training data (X_train) for constraints.
# The train set is huge; we'll load it separately or just use test set bounds
# for this demo. Let's load a small slice of train to get reasonable mins/maxs.
X_train = data['X_train']
train_mins = X_train.min(axis=0)
train_maxs = X_train.max(axis=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_train.shape[1]
model = NIDSModel(input_dim).to(device)
model.load_state_dict(torch.load('models/baseline_nids_full.pth', map_location=device))
model.eval()

print(f"Model loaded. Test set: {X_test.shape}")

# For speed, we'll use a subset of test attacks for black-box GA (1000 samples max)
attack_indices = np.where(y_test == 1)[0][:200]  # 200 attack samples
X_attack = X_test[attack_indices]
y_attack = y_test[attack_indices]

# Clean accuracy on the full test set
X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

with torch.no_grad():
    clean_preds = torch.argmax(model(X_test_t), dim=1).cpu().numpy()
clean_acc = accuracy_score(y_test, clean_preds)
print(f"Clean accuracy (full test): {clean_acc:.4f}")

# ------------------------------------------------------------------
# 2. Constrained FGSM attack
# ------------------------------------------------------------------
def fgsm_constrained(model, inputs, labels, epsilon, mins_t, maxs_t):
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    pert = epsilon * inputs.grad.sign()
    adv = inputs + pert
    adv = torch.max(adv, mins_t)
    adv = torch.min(adv, maxs_t)
    return adv

# ------------------------------------------------------------------
# 3. Constrained PGD attack
# ------------------------------------------------------------------
def pgd_constrained(model, inputs, labels, epsilon, alpha, iters, mins_t, maxs_t):
    original = inputs.clone().detach()
    adv = inputs.clone().detach()
    for _ in range(iters):
        adv.requires_grad = True
        outputs = model(adv)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            eta = torch.clamp(adv - original, -epsilon, epsilon)
            adv = original + eta
            adv = torch.max(adv, mins_t)
            adv = torch.min(adv, maxs_t)
    return adv

# ------------------------------------------------------------------
# 4. Black-box Genetic Algorithm attack (fitness = benign prob)
# ------------------------------------------------------------------
def fitness(model, individuals):
    indiv_t = torch.tensor(individuals, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(indiv_t)
        probs = F.softmax(logits, dim=1)
    return probs[:, 0].cpu().numpy()   # p(BENIGN)

def genetic_attack(model, sample, mins, maxs, pop_size=60, generations=150,
                   mutation_rate=0.2, sigma=0.05):
    n_feat = len(sample)
    pop = np.tile(sample, (pop_size, 1)) + np.random.normal(0, sigma, (pop_size, n_feat))
    pop = np.clip(pop, mins, maxs)
    best_adv = sample.copy()
    best_fit = fitness(model, sample.reshape(1, -1))[0]
    for gen in range(generations):
        fit = fitness(model, pop)
        gen_best_idx = np.argmax(fit)
        if fit[gen_best_idx] > best_fit:
            best_fit = fit[gen_best_idx]
            best_adv = pop[gen_best_idx].copy()
        if best_fit > 0.5:
            return best_adv, True
        # Elite selection (top 1/6)
        elite_idx = np.argsort(fit)[-max(1, pop_size//6):]
        new_pop = pop[elite_idx].copy()
        while len(new_pop) < pop_size:
            p1, p2 = np.random.choice(pop_size, 2, replace=False)
            parent1, parent2 = pop[p1], pop[p2]
            if np.random.rand() < 0.5:
                point = np.random.randint(1, n_feat)
                child = np.concatenate([parent1[:point], parent2[point:]])
            else:
                child = parent1.copy()
            mask = np.random.rand(n_feat) < mutation_rate
            child = child + np.random.normal(0, sigma, n_feat) * mask
            child = np.clip(child, mins, maxs)
            new_pop = np.vstack([new_pop, child.reshape(1, -1)])
        pop = new_pop[:pop_size]
    return best_adv, best_fit > 0.5

# ------------------------------------------------------------------
# 5. Run all attacks and measure
# ------------------------------------------------------------------
mins_t = torch.tensor(train_mins, dtype=torch.float32, device=device)
maxs_t = torch.tensor(train_maxs, dtype=torch.float32, device=device)

# FGSM with epsilon=1.0
print("\nRunning constrained FGSM (ε=1.0)...")
start = time.time()
adv_fgsm = fgsm_constrained(model, X_test_t, y_test_t, 1.0, mins_t, maxs_t)
with torch.no_grad():
    fgsm_preds = torch.argmax(model(adv_fgsm), dim=1).cpu().numpy()
fgsm_time = time.time() - start
attack_mask = (y_test == 1)
fgsm_success = ((fgsm_preds == 0) & attack_mask).sum() / attack_mask.sum()
fgsm_acc = accuracy_score(y_test, fgsm_preds)
print(f"  FGSM: success={fgsm_success:.4f}, accuracy={fgsm_acc:.4f}, time={fgsm_time:.1f}s")

# PGD with epsilon=1.0, alpha=0.1, iters=20
print("Running constrained PGD (ε=1.0, iters=20)...")
start = time.time()
adv_pgd = pgd_constrained(model, X_test_t, y_test_t, 1.0, 0.1, 20, mins_t, maxs_t)
with torch.no_grad():
    pgd_preds = torch.argmax(model(adv_pgd), dim=1).cpu().numpy()
pgd_time = time.time() - start
pgd_success = ((pgd_preds == 0) & attack_mask).sum() / attack_mask.sum()
pgd_acc = accuracy_score(y_test, pgd_preds)
print(f"  PGD: success={pgd_success:.4f}, accuracy={pgd_acc:.4f}, time={pgd_time:.1f}s")

# GA on 200 attack samples (150 gen, pop 60)
print("Running black-box GA (200 samples, 150 generations)...")
start = time.time()
ga_successes = 0
ga_preds_list = []
for idx, sample in enumerate(X_attack):
    adv, success = genetic_attack(model, sample, train_mins, train_maxs,
                                  pop_size=60, generations=150)
    ga_successes += success
    adv_t = torch.tensor(adv, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(adv_t)
        pred = torch.argmax(logits, dim=1).cpu().item()
    ga_preds_list.append(pred)
    if (idx+1) % 50 == 0:
        print(f"  GA processed {idx+1}/{len(X_attack)}...")
ga_time = time.time() - start
ga_success_rate = ga_successes / len(X_attack)
ga_preds = np.array(ga_preds_list)
ga_acc = accuracy_score(y_attack, ga_preds)
print(f"  GA: success={ga_success_rate:.4f}, accuracy={ga_acc:.4f}, time={ga_time:.1f}s")

# ------------------------------------------------------------------
# 6. Save summary and plot
# ------------------------------------------------------------------
attacks = ['FGSM (ε=1.0)', 'PGD (ε=1.0)', f'GA (150 gen)']
success_rates = [fgsm_success, pgd_success, ga_success_rate]
accuracies = [fgsm_acc, pgd_acc, ga_acc]

with open('results/comparison_full.txt', 'w') as f:
    f.write("Adversarial Attack Comparison on Full CIC-IDS2017 NIDS\n")
    f.write("="*55 + "\n")
    f.write(f"{'Attack':<20} {'Success Rate':<15} {'Model Accuracy':<15} {'Time (s)':<10}\n")
    f.write("-"*55 + "\n")
    for a, s, ac, t in zip(attacks, success_rates, accuracies,
                           [fgsm_time, pgd_time, ga_time]):
        f.write(f"{a:<20} {s:<15.2f} {ac:<15.4f} {t:<10.1f}\n")
    f.write("\nGA evaluated on 200 attack samples due to runtime.\n")

x = np.arange(len(attacks))
width = 0.35
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, success_rates, width, label='Attack Success Rate')
bars2 = ax.bar(x + width/2, accuracies, width, label='Model Accuracy')
ax.set_ylabel('Rate')
ax.set_title('NIDS Adversarial Evasion (Full CIC-IDS2017)')
ax.set_xticks(x)
ax.set_xticklabels(attacks)
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig('results/comparison_full.png')
print("\nSummary saved to results/comparison_full.txt and .png")
