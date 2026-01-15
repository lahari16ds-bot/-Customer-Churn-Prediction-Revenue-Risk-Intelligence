import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load customer behavior data
data = pd.read_csv("customer_data.csv")

# Features for churn prediction
X = data[[
    "days_since_last_purchase",
    "total_orders",
    "average_order_value",
    "lifetime_value"
]]

# Target variable (1 = churned, 0 = active)
y = data["churn"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Predict churn probabilities for all customers
data["churn_probability"] = model.predict_proba(X)[:, 1]

# Save output for Excel & Tableau
data.to_csv("churn_predictions.csv", index=False)

print("Churn prediction file created: churn_predictions.csv")
