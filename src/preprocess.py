import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Correct path to the Friday DDoS CSV (nested inside MachineLearningCSV/MachineLearningCVE/)
df = pd.read_csv('data/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Clean infinity/NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Binary label: 0 = BENIGN, 1 = attack
# Note: the column name has a leading space: ' Label'
df['is_attack'] = df[' Label'].apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
y = df['is_attack'].values

# Select only numeric features, drop columns with only one unique value
X = df.select_dtypes(include=[np.number])
X = X.loc[:, X.nunique() > 1]

# Train/test split (stratified to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize (fit only on train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save everything for later use
np.savez_compressed('data/ids2017_ddos_test.npz',
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test)

print(f"Preprocessing complete. Train: {X_train.shape}, Test: {X_test.shape}")
