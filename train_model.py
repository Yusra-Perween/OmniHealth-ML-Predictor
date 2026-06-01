import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Step 1: Load data
data = pd.read_csv("health_data.csv")  # real dataset

# Step 2: Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Save trained model
dump(model, "health_model.pkl")

print("✅ Model trained on real data and saved as health_model.pkl")
