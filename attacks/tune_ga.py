import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import NIDSModel
import joblib

# Load data and model (same as before)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_train.shape[1]
model = NIDSModel(input_dim).to(device)
model.load_state_dict(torch.load('models/baseline_nids.pth', map_location=device))
model.eval()

train_mins = X_train.min(axis=0)
train_maxs = X_train.max(axis=0)

# Pick one attack sample
attack_idx = np.where(y_test == 1)[0][0]
sample = X_test[attack_idx]

def fitness(indiv):
    indiv_t = torch.tensor(indiv, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(indiv_t)
        probs = F.softmax(logits, dim=1)
    return probs[:, 0].cpu().numpy()

def genetic_attack_fixed_params(sample, mins, maxs, pop_size, generations, mutation_rate=0.2, sigma=0.05):
    n_features = len(sample)
    population = np.tile(sample, (pop_size, 1)) + np.random.normal(0, sigma, (pop_size, n_features))
    population = np.clip(population, mins, maxs)
    best_adv = sample.copy()
    best_fit = fitness(sample.reshape(1, -1))[0]
    for gen in range(generations):
        fit = fitness(population)
        gen_best_idx = np.argmax(fit)
        if fit[gen_best_idx] > best_fit:
            best_fit = fit[gen_best_idx]
            best_adv = population[gen_best_idx].copy()
        if best_fit > 0.5:
            return best_adv, True, gen+1  # success, return generation count
        elite_indices = np.argsort(fit)[-max(1, pop_size//6):]  # top 1/6
        new_pop = population[elite_indices].copy()
        while len(new_pop) < pop_size:
            p1_idx, p2_idx = np.random.choice(pop_size, 2, replace=False)
            parent1, parent2 = population[p1_idx], population[p2_idx]
            if np.random.rand() < 0.5:
                point = np.random.randint(1, n_features)
                child = np.concatenate([parent1[:point], parent2[point:]])
            else:
                child = parent1.copy()
            mask = np.random.rand(n_features) < mutation_rate
            child = child + np.random.normal(0, sigma, n_features) * mask
            child = np.clip(child, mins, maxs)
            new_pop = np.vstack([new_pop, child.reshape(1, -1)])
        population = new_pop[:pop_size]
    return best_adv, best_fit > 0.5, generations

# Parameter grid
pop_sizes = [30, 60, 100]
generations_list = [40, 80, 150]
print("Tuning GA parameters on one sample:")
print("Pop_Size | Generations | Success | Generations to success")
for pop in pop_sizes:
    for gen in generations_list:
        adv, success, gens_needed = genetic_attack_fixed_params(sample, train_mins, train_maxs, pop, gen)
        print(f" {pop:7d} | {gen:11d} | {str(success):7s} | {gens_needed if success else 'N/A'}")
