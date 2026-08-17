import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

# Separate features and target
X = data.drop("Label", axis=1)
y = data["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\n========== TRAIN/TEST SPLIT ==========")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\n========== MODEL TRAINING ==========")
print("Random Forest training completed!")

# Make predictions on test data
predictions = model.predict(X_test)

print("\n========== PREDICTIONS ==========")
print("Actual labels:   ", y_test.to_numpy())
print("Predicted labels:", predictions)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL ACCURACY ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, predictions)

print("\n========== CONFUSION MATRIX ==========")
print(cm)