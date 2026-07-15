import pandas as pd
import numpy as np

# Load the same CSV file (not scaled)
df = pd.read_csv('data/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Use the same numeric columns as preprocess.py (they match by index after we select dtypes)
X_raw = df.select_dtypes(include=[np.number])
# Also drop the zero-variance columns (we'll need to match exactly what preprocess did)
X_raw = X_raw.loc[:, X_raw.nunique() > 1]

# Record min and max for each feature in the raw space
mins = X_raw.min().values
maxs = X_raw.max().values
feature_names = X_raw.columns.tolist()

# Save these constraints
np.savez('data/feature_constraints.npz', mins=mins, maxs=maxs, columns=feature_names)
print(f"Constraints saved. Features: {len(mins)}")
print("Sample min values:", mins[:5])
print("Sample max values:", maxs[:5])

