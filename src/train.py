import pandas as pd

# Load dataset
data = pd.read_csv("data/sample/sample_features.csv")

print("========== DATASET ==========")
print(data)

print("\n========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== SHAPE ==========")
print(data.shape)

print("\n========== COLUMN NAMES ==========")
print(data.columns)

print("\n========== DATA TYPES ==========")
print(data.dtypes)