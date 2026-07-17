import pandas as pd

# Load dataset
data = pd.read_csv("data/sample/sample_features.csv")

#Display dataset
print("========== DATASET ==========")
print(data)

# Separate Features (X) and Label (y)
X = data.drop("Label",axis=1)
y=data["Label"]

print("\n========== FEATURES (X) ==========")
print(X)

print("\n========== LABELS (y) ==========")
print(y)