import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("Loading dataset...")
df = pd.read_csv(
    "energy_data.csv",
    sep=";",
    low_memory=False
)

print("Original columns:")
print(df.columns.tolist())

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

# Keep only useful columns
required_columns = [
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

df = df[required_columns].copy()

# Convert numeric columns
numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing values
df.dropna(inplace=True)

print(f"Rows after cleaning: {len(df)}")

# Use a sample to make training faster on normal laptops
if len(df) > 50000:
    df = df.sample(50000, random_state=42)
    print("Using sample of 50000 rows for faster training.")

# Create datetime column
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

df.dropna(subset=["datetime"], inplace=True)

# Extract time-based features
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year
df["day_of_week"] = df["datetime"].dt.dayofweek
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute

# Features and target
feature_columns = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "day",
    "month",
    "year",
    "day_of_week",
    "hour",
    "minute"
]

target_column = "Global_active_power"

X = df[feature_columns]
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Performance")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


lr_metrics = evaluate_model("Linear Regression", y_test, lr_preds)
rf_metrics = evaluate_model("Random Forest", y_test, rf_preds)

# Select best model
if rf_metrics["R2"] >= lr_metrics["R2"]:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_preds = rf_preds
else:
    best_model = lr_model
    best_model_name = "Linear Regression"
    best_preds = lr_preds

# Save model and feature names
joblib.dump(best_model, "model.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print(f"\nBest model saved: {best_model_name}")
print("Saved files: model.pkl, feature_columns.pkl")

# Save metrics to file
with open("model_results.txt", "w", encoding="utf-8") as f:
    f.write("Energy Consumption Prediction Results\n")
    f.write("=" * 40 + "\n\n")
    f.write("Linear Regression\n")
    for k, v in lr_metrics.items():
        f.write(f"{k}: {v:.4f}\n")

    f.write("\nRandom Forest\n")
    for k, v in rf_metrics.items():
        f.write(f"{k}: {v:.4f}\n")

    f.write(f"\nBest Model: {best_model_name}\n")

# Plot 1: Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(best_preds[:200], label="Predicted")
plt.title(f"Actual vs Predicted Energy Consumption ({best_model_name})")
plt.xlabel("Sample Index")
plt.ylabel("Global Active Power")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# Plot 2: Model comparison
model_names = ["Linear Regression", "Random Forest"]
r2_scores = [lr_metrics["R2"], rf_metrics["R2"]]

plt.figure(figsize=(8, 5))
plt.bar(model_names, r2_scores)
plt.title("Model Comparison (R2 Score)")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# Plot 3: Feature importance for Random Forest
if hasattr(rf_model, "feature_importances_"):
    importances = rf_model.feature_importances_

    plt.figure(figsize=(10, 5))
    plt.bar(feature_columns, importances)
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

print("\nTraining completed successfully.")
