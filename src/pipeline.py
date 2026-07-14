#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
================================================================================
END-TO-END ML PIPELINE: California Housing Price Prediction
================================================================================
Follows the full workflow:
1. Understand the Problem
2. Collect Data
3. Explore the Data (EDA)
4. Clean & Preprocess Data
5. Feature Engineering
6. Split Data (Train/Validation/Test)
7. Build a Baseline Model
8. Train Multiple Algorithms
9. Evaluate Models
10. Hyperparameter Tuning
11. Cross Validation
12. Handle Overfitting/Underfitting
13. Feature Selection
14. Build Final Pipeline
15. Save Model (Pickle/Joblib)
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, saves plots to disk
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
OUTPUT_DIR = "."

# ==============================================================================
# 1. UNDERSTAND THE PROBLEM
# ==============================================================================
# Goal: Predict median house value (in $100,000s) for California districts
# based on features like income, house age, rooms, population, location, etc.
# This is a SUPERVISED REGRESSION problem.
print("=" * 80)
print("STEP 1: UNDERSTAND THE PROBLEM")
print("=" * 80)
print("""
Task type   : Regression
Target      : MedHouseVal (median house value, in $100,000 units)
Features    : 8 numeric predictors (income, age, rooms, location, etc.)
Success     : Minimize RMSE / MAE, maximize R^2 on unseen data
""")

# ==============================================================================
# 2. COLLECT DATA
# ==============================================================================
print("=" * 80)
print("STEP 2: COLLECT DATA")
print("=" * 80)
housing = fetch_california_housing(as_frame=True)
df = housing.frame  # includes target column 'MedHouseVal'
print(f"Shape: {df.shape}")
print(df.head())

# ==============================================================================
# 3. EXPLORE THE DATA (EDA)
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 3: EXPLORE THE DATA (EDA)")
print("=" * 80)
print("\n--- Info ---")
df.info()
print("\n--- Describe ---")
print(df.describe())
print("\n--- Missing values ---")
print(df.isnull().sum())
print("\n--- Correlation with target ---")
print(df.corr()["MedHouseVal"].sort_values(ascending=False))

# Save a few exploratory plots
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_correlation_heatmap.png", dpi=100)
plt.close()

df.hist(bins=50, figsize=(14, 10))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_histograms.png", dpi=100)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"],
            cmap="viridis", s=df["Population"] / 100, alpha=0.4)
plt.colorbar(label="Median House Value")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Geographic Distribution of House Values")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_geo_scatter.png", dpi=100)
plt.close()
print("Saved EDA plots: eda_correlation_heatmap.png, eda_histograms.png, eda_geo_scatter.png")

# ==============================================================================
# 4. CLEAN & PREPROCESS DATA
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 4: CLEAN & PREPROCESS DATA")
print("=" * 80)
# The sklearn California dataset has no missing values or duplicates by
# default, but we handle it robustly anyway (as you would with raw/real data).
print(f"Duplicate rows: {df.duplicated().sum()}")
df = df.drop_duplicates()

# Cap extreme outliers in target (common with this dataset: capped at 5.0)
capped_at_max = (df["MedHouseVal"] >= 5.0).sum()
print(f"Rows with capped target value (>=5.0): {capped_at_max}")
# Option: remove capped rows to avoid biasing the model (kept here, but flagged)

# Simple imputer safeguard for numeric columns (in case of NaNs in future data)
numeric_features = housing.feature_names  # the 8 original features
imputer = SimpleImputer(strategy="median")
df[numeric_features] = imputer.fit_transform(df[numeric_features])
print("Missing values handled with median imputation (safeguard).")

# ==============================================================================
# 5. FEATURE ENGINEERING
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 5: FEATURE ENGINEERING")
print("=" * 80)
df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"].replace(0, np.nan)
df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"].replace(0, np.nan)
df["PopulationPerHousehold"] = df["Population"] / df["AveOccup"].replace(0, np.nan)
df = df.fillna(df.median(numeric_only=True))
print("Added engineered features: RoomsPerHousehold, BedroomsPerRoom, PopulationPerHousehold")
print(df[["RoomsPerHousehold", "BedroomsPerRoom", "PopulationPerHousehold"]].describe())

TARGET = "MedHouseVal"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ==============================================================================
# 6. SPLIT DATA (TRAIN / VALIDATION / TEST)
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 6: SPLIT DATA (TRAIN/VALIDATION/TEST)")
print("=" * 80)
# 60% train, 20% validation, 20% test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE  # 0.25*0.8=0.2
)
print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# ==============================================================================
# 7. BUILD A BASELINE MODEL
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 7: BUILD A BASELINE MODEL")
print("=" * 80)
scaler_base = StandardScaler()
X_train_scaled_base = scaler_base.fit_transform(X_train)
X_val_scaled_base = scaler_base.transform(X_val)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train_scaled_base, y_train)
baseline_pred = baseline.predict(X_val_scaled_base)
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
print(f"Baseline (predict mean) RMSE on validation: {baseline_rmse:.4f}")
print("Any real model must beat this to be considered useful.")

# ==============================================================================
# 8. TRAIN MULTIPLE ALGORITHMS
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 8: TRAIN MULTIPLE ALGORITHMS")
print("=" * 80)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

candidate_models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "SVR": SVR(),
}

results = {}
for name, model in candidate_models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"{name:18s} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

# ==============================================================================
# 9. EVALUATE MODELS
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 9: EVALUATE MODELS")
print("=" * 80)
results_df = pd.DataFrame(results).T.sort_values("RMSE")
print(results_df)
best_model_name = results_df.index[0]
print(f"\nBest performing model on validation set: {best_model_name}")

# ==============================================================================
# 10. HYPERPARAMETER TUNING
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 10: HYPERPARAMETER TUNING")
print("=" * 80)
# Tune the RandomForest (typically a strong, robust performer for this dataset)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train_scaled, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")
tuned_model = grid_search.best_estimator_

# ==============================================================================
# 11. CROSS VALIDATION
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 11: CROSS VALIDATION")
print("=" * 80)
cv_scores = cross_val_score(
    tuned_model, X_train_scaled, y_train, cv=5, scoring="neg_root_mean_squared_error"
)
cv_rmse = -cv_scores
print(f"5-fold CV RMSE scores: {np.round(cv_rmse, 4)}")
print(f"Mean: {cv_rmse.mean():.4f} | Std: {cv_rmse.std():.4f}")

# ==============================================================================
# 12. HANDLE OVERFITTING / UNDERFITTING
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 12: HANDLE OVERFITTING/UNDERFITTING")
print("=" * 80)
train_pred = tuned_model.predict(X_train_scaled)
val_pred = tuned_model.predict(X_val_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
gap = val_rmse - train_rmse
if gap > 0.15:
    print(f"Gap = {gap:.4f} -> model may be OVERFITTING (consider regularization, "
          f"fewer trees/depth, more data, or simpler model).")
elif train_rmse > baseline_rmse * 0.9:
    print(f"Train error close to baseline -> model may be UNDERFITTING "
          f"(consider more complex model or better features).")
else:
    print(f"Gap = {gap:.4f} -> reasonable fit, no strong sign of over/underfitting.")

# Learning curve to visualize
train_sizes, train_scores, val_scores = learning_curve(
    tuned_model, X_train_scaled, y_train, cv=5,
    scoring="neg_root_mean_squared_error",
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, -train_scores.mean(axis=1), "o-", label="Training RMSE")
plt.plot(train_sizes, -val_scores.mean(axis=1), "o-", label="Validation RMSE")
plt.xlabel("Training Set Size"); plt.ylabel("RMSE")
plt.title("Learning Curve")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/learning_curve.png", dpi=100)
plt.close()
print("Saved learning_curve.png")

# ==============================================================================
# 13. FEATURE SELECTION
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 13: FEATURE SELECTION")
print("=" * 80)
feature_importances = pd.Series(
    tuned_model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("Feature importances:")
print(feature_importances)

selector = SelectFromModel(tuned_model, threshold="median", prefit=True)
selected_features = X.columns[selector.get_support()]
print(f"\nSelected {len(selected_features)} / {len(X.columns)} features "
      f"(importance >= median):")
print(list(selected_features))

plt.figure(figsize=(8, 6))
feature_importances.plot(kind="barh")
plt.title("Feature Importances (Tuned Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importances.png", dpi=100)
plt.close()
print("Saved feature_importances.png")

# ==============================================================================
# 14. BUILD FINAL PIPELINE
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 14: BUILD FINAL PIPELINE")
print("=" * 80)
# A single sklearn Pipeline bundling preprocessing + final tuned model,
# so it can be applied to raw incoming data with one .predict() call.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, list(X.columns)),
])
final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(**grid_search.best_params_, random_state=RANDOM_STATE, n_jobs=-1)),
])

# Refit on train+validation for a final model, evaluate once on untouched test set
final_pipeline.fit(X_train_full, y_train_full)
test_pred = final_pipeline.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)
print(f"FINAL TEST SET PERFORMANCE:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE : {test_mae:.4f}")
print(f"  R2  : {test_r2:.4f}")

# ==============================================================================
# 15. SAVE MODEL (PICKLE / JOBLIB)
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 15: SAVE MODEL")
print("=" * 80)
joblib.dump(final_pipeline, f"{OUTPUT_DIR}/california_housing_pipeline.joblib")
print("Saved pipeline to california_housing_pipeline.joblib")

# Example of loading it back and predicting on a new sample
loaded_pipeline = joblib.load(f"{OUTPUT_DIR}/california_housing_pipeline.joblib")
sample = X_test.iloc[[0]]
print(f"\nExample prediction on a test sample:")
print(f"  Predicted: {loaded_pipeline.predict(sample)[0]:.4f}  |  Actual: {y_test.iloc[0]:.4f}")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)


# In[ ]:





# In[ ]:





# In[ ]:







# In[ ]:




