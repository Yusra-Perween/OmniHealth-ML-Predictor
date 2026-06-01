import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("heart.csv")

# Drop unnecessary columns if present
if 'id' in df.columns:
    df = df.drop(['id', 'dataset'], axis=1)

# Encode categorical columns automatically
label_enc = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_enc.fit_transform(df[col])

# Prepare features and target
X = df.drop('num', axis=1)
y = df['num']

# Convert target to binary: 0 = no disease, 1 = disease
y = y.apply(lambda x: 1 if x > 0 else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", accuracy)

# Save model
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ heart_model.pkl saved successfully.")
