import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from model import NIDSModel
import joblib

# ------------------------------------------------------------------
# 1. Load data and model (same as before)
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
joblib.dump(scaler, 'models/scaler.pkl')

train_mins = X_train.min(axis=0)
train_maxs = X_train.max(axis=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_train.shape[1]
model = NIDSModel(input_dim).to(device)
model.load_state_dict(torch.load('models/baseline_nids.pth', map_location=device))
model.eval()

# We'll attack a subset of true attack samples to keep run-time reasonable
attack_indices = np.where(y_test == 1)[0][:100]  # first 100 true attacks
X_attack = X_test[attack_indices]                 # clean attack flows

# Convert to tensor for model evaluation
X_attack_t = torch.tensor(X_attack, dtype=torch.float32, device=device)

# ------------------------------------------------------------------
# 2. Genetic algorithm parameters
# ------------------------------------------------------------------
POP_SIZE = 30          # number of individuals in each generation
GENERATIONS = 40       # max generations
MUTATION_RATE = 0.2    # probability of mutating each feature
CROSSOVER_RATE = 0.5   # probability of crossover vs. direct copy
SIGMA = 0.05           # standard deviation of Gaussian mutation noise
ELITE_SIZE = 5         # top individuals carried to next generation unchanged
THRESHOLD = 0.5        # if predicted benign probability > 0.5, success

# ------------------------------------------------------------------
# 3. Helper: fitness = probability that the model says "benign"
# ------------------------------------------------------------------
def fitness(individuals):
    """individuals: numpy array of shape (n, n_features), already scaled"""
    indiv_t = torch.tensor(individuals, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(indiv_t)                # raw scores
        probs = F.softmax(logits, dim=1)       # [p_benign, p_attack]
        benign_probs = probs[:, 0].cpu().numpy()  # index 0 = BENIGN
    return benign_probs   # higher = more benign

# ------------------------------------------------------------------
# 4. Genetic attack on a single clean attack sample
# ------------------------------------------------------------------
def genetic_attack(sample, mins, maxs):
    """
    sample: clean attack features (scaled numpy array, shape (n_features,))
    mins, maxs: feature bounds (scaled) per feature
    Returns: adversarial example (scaled) and whether attack succeeded.
    """
    n_features = len(sample)
    # Initialize population: clean sample + small Gaussian noise, clipped to bounds
    population = np.tile(sample, (POP_SIZE, 1)) + np.random.normal(0, SIGMA, (POP_SIZE, n_features))
    population = np.clip(population, mins, maxs)

    best_adv = sample.copy()
    best_fitness = fitness(sample.reshape(1, -1))[0]

    for gen in range(GENERATIONS):
        # Evaluate fitness for entire population
        fit = fitness(population)  # shape (POP_SIZE,)

        # Find best in this generation
        gen_best_idx = np.argmax(fit)
        gen_best_fit = fit[gen_best_idx]
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_adv = population[gen_best_idx].copy()
        # Early stop if we've reached benign threshold
        if best_fitness > THRESHOLD:
            break

        # Selection: pick top ELITE_SIZE, then fill rest by tournament selection
        elite_indices = np.argsort(fit)[-ELITE_SIZE:]
        new_population = population[elite_indices].copy()

        while len(new_population) < POP_SIZE:
            # Tournament selection: pick 3 random individuals, keep the fittest
            candidates = np.random.choice(POP_SIZE, 3, replace=False)
            parent1_idx = candidates[np.argmax(fit[candidates])]
            candidates = np.random.choice(POP_SIZE, 3, replace=False)
            parent2_idx = candidates[np.argmax(fit[candidates])]
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover with probability CROSSOVER_RATE
            if np.random.rand() < CROSSOVER_RATE:
                # single-point crossover
                point = np.random.randint(1, n_features)
                child = np.concatenate([parent1[:point], parent2[point:]])
            else:
                child = parent1.copy()  # or parent2, either works

            # Mutation: add Gaussian noise with probability MUTATION_RATE per feature
            mutation_mask = np.random.rand(n_features) < MUTATION_RATE
            noise = np.random.normal(0, SIGMA, n_features) * mutation_mask
            child = child + noise

            # Clip to valid bounds
            child = np.clip(child, mins, maxs)
            new_population = np.vstack([new_population, child.reshape(1, -1)])

        population = new_population[:POP_SIZE]  # ensure exact size

    return best_adv, best_fitness > THRESHOLD

# ------------------------------------------------------------------
# 5. Evaluate on the 100 attack samples
# ------------------------------------------------------------------
print("Running black-box genetic attacks on 100 true attack samples...")
successes = 0
adversarial_examples = []
for idx, sample in enumerate(X_attack):
    adv, success = genetic_attack(sample, train_mins, train_maxs)
    if success:
        successes += 1
    adversarial_examples.append(adv)
    if (idx+1) % 20 == 0:
        print(f"  Processed {idx+1}/{len(X_attack)}...")

# Compute overall success rate and accuracy drop
adv_array = np.array(adversarial_examples)
adv_t = torch.tensor(adv_array, dtype=torch.float32, device=device)
with torch.no_grad():
    adv_preds = torch.argmax(model(adv_t), dim=1).cpu().numpy()
# Original labels for these 100 samples are all 1 (attack)
original_labels = np.ones(len(X_attack))
# Success rate among attacks (all were attacks)
ga_success_rate = successes / len(X_attack)
# Overall accuracy on this subset (should be low if attack works)
ga_accuracy = accuracy_score(original_labels, adv_preds)

print(f"\nGA Black-Box Results on 100 attack samples:")
print(f"  Attack success rate (benign probability > 0.5): {ga_success_rate:.4f}")
print(f"  Accuracy of the model on adversarial samples: {ga_accuracy:.4f}")

# Also evaluate on the full test set if you wish (optional)
# We'll just report the subset.
