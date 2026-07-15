import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import os

# ------------------------------------------------------------
# 1. Load all CSV files from the full CIC-IDS2017 (CSV folder)
# ------------------------------------------------------------
data_dir = 'data/MachineLearningCSV/MachineLearningCVE/'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and not f.startswith('._')]

df_list = []
for f in csv_files:
    path = os.path.join(data_dir, f)
    print(f"Loading {f}...")
    df = pd.read_csv(path)
    # Standardize column names: strip leading/trailing spaces
    df.columns = df.columns.str.strip()
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)
print(f"Total rows before cleaning: {len(full_df)}")

# ------------------------------------------------------------
# 2. Clean: infinities, NaNs, and label
# ------------------------------------------------------------
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

# The label column is now 'Label' (after stripping whitespace)
full_df['is_attack'] = full_df['Label'].apply(
    lambda x: 0 if x.strip().upper() == 'BENIGN' else 1)
y = full_df['is_attack'].values

# Keep only numeric features, drop zero-variance columns
X_raw = full_df.select_dtypes(include=[np.number])
X_raw = X_raw.loc[:, X_raw.nunique() > 1]

print(f"Cleaned dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")

# ------------------------------------------------------------
# 3. Balance: downsample benign class to match attack count
# ------------------------------------------------------------
# Separate benign and attack indices
benign_idx = np.where(y == 0)[0]
attack_idx = np.where(y == 1)[0]
print(f"Original class counts: Benign={len(benign_idx)}, Attack={len(attack_idx)}")

if len(benign_idx) > len(attack_idx):
    # Downsample benign to same number as attacks
    benign_down = resample(benign_idx, n_samples=len(attack_idx),
                           random_state=42, replace=False)
    balanced_idx = np.concatenate([benign_down, attack_idx])
else:
    balanced_idx = np.arange(len(y))  # if attacks are more (unlikely), keep all

np.random.shuffle(balanced_idx)
X_bal = X_raw.iloc[balanced_idx].values
y_bal = y[balanced_idx]
print(f"Balanced dataset: {len(X_bal)} samples (Benign={np.sum(y_bal==0)}, Attack={np.sum(y_bal==1)})")

# ------------------------------------------------------------
# 4. Train/test split (stratified)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

# ------------------------------------------------------------
# 5. Scale and save
# ------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

np.savez_compressed('data/ids2017_full_balanced.npz',
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test)
import joblib
joblib.dump(scaler, 'models/scaler_full.pkl')
print(f"Saved full balanced dataset. Train: {X_train.shape}, Test: {X_test.shape}")
