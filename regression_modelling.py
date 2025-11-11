import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

bio = pd.read_excel("Data/biomarkers.xlsx")
cov = pd.read_excel("Data/covariates.xlsx")

bio_time_0 = bio[bio["Biomarker"].str.contains("0weeks")].copy()
bio_time_0["PatientID"] = bio_time_0["Biomarker"].str.extract(r"(\d+)").astype(int)

merged_datasets = pd.merge(bio_time_0, cov, on="PatientID", how="inner")

# dropping the rows with missing 12-month VAS scores
merged_datasets = merged_datasets.dropna(subset=["Vas-12months"])

biomarkers = [
    "IL-8", "VEGF-A", "OPG", "TGF-beta-1", "IL-6", "CXCL9", "CXCL1", "IL-18", "CSF-1"
]

# keep only numeric covariates automatically, plus VAS-at-inclusion
numeric_covs = merged_datasets.select_dtypes(include=["number"]).columns.tolist()

# removing identifiers and VAS-12months, which is the target variable, from predictors
remove_cols = ["PatientID", "Vas-12months"]

X_cols = [c for c in numeric_covs if c not in remove_cols]

X = merged_datasets[X_cols]
y = merged_datasets["Vas-12months"]

# splitting data into training (80 %) and test (20 %) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)

print(f"\nTraining set size: {X_train.shape[0]} patients")
print(f"Test set size: {X_test.shape[0]} patients\n")


model = LinearRegression()
model.fit(X_train, y_train)

coef_table = pd.DataFrame({
    "Predictor": ["Intercept"] + list(X_train.columns),
    "Coefficient": [model.intercept_] + list(model.coef_)
})

print("\nMultiple Linear Regression Model: Predicting 12-month VAS\n")
print(coef_table.to_string(index=False))

y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)

print("\nModel Fit (Training set):")
print(f"R² = {r2_train}")

# out of-sample evaluation on the test set
y_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_pred)

print("\nOut-of-Sample Evaluation (Test set):")
print(f"R² = {r2_test}")

comparison = pd.DataFrame({
    "Actual_VAS_12m": y_test,
    "Predicted_VAS_12m": y_pred
})
print("\nPredicted vs Actual (Test set):\n")
print(comparison.to_string(index=False))

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, edgecolor='k')

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2, label='Perfect prediction (y = x)')

plt.xlabel("Actual VAS-score at 12-months", fontsize=12)
plt.ylabel("Predicted VAS-score at 12-months", fontsize=12)
plt.title("Predicted vs Actual VAS Scores at 12-months for Test Data", fontsize=13)
plt.grid(True, linestyle=':', alpha=0.8)
plt.legend()
plt.tight_layout()
os.makedirs("Visualisations", exist_ok=True)
plt.savefig("Visualisations/predicted_vs_actual_vas_scores_at_12_months.png")
plt.show()

