import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("data/sample/sample_features.csv")

# Create encoders
stage_encoder = LabelEncoder()
label_encoder = LabelEncoder()

# Encode Stage column
data["Stage"] = stage_encoder.fit_transform(data["Stage"])

# Encode Label column
data["Label"] = label_encoder.fit_transform(data["Label"])

print("========== ENCODED DATASET ==========")
print(data)

print("\nStage Mapping:")
for i, name in enumerate(stage_encoder.classes_):
    print(f"{name} -> {i}")

print("\nLabel Mapping:")
for i, name in enumerate(label_encoder.classes_):
    print(f"{name} -> {i}")