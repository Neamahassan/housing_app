"""
train_model.py
--------------
Trains and compares several regression models on the Boston housing data,
picks the best one by cross-validation, and saves everything the Streamlit
app needs (model.joblib + metrics.json).

Run:  python train_model.py
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, ShuffleSplit,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

SEED = 42
FEATURES = ["RM", "LSTAT", "PTRATIO"]
TARGET = "MEDV"

# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
data = pd.read_csv("housing.csv")
X, y = data[FEATURES], data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"Rows: {len(data)} | train: {len(X_train)} | test: {len(X_test)}")

# ----------------------------------------------------------------------------
# 2. Compare candidate models with cross-validation (training data only)
# ----------------------------------------------------------------------------
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

candidates = {
    "Linear Regression": LinearRegression(),
    "Decision Tree (depth tuned)": GridSearchCV(
        DecisionTreeRegressor(random_state=SEED),
        {"max_depth": list(range(1, 11))}, cv=cv, scoring="r2"),
    "Random Forest": GridSearchCV(
        RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        {"max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 3, 5]},
        cv=cv, scoring="r2"),
    "Gradient Boosting": GridSearchCV(
        GradientBoostingRegressor(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
        cv=cv, scoring="r2"),
}

comparison = []
fitted = {}
for name, est in candidates.items():
    est.fit(X_train, y_train)
    best = est.best_estimator_ if hasattr(est, "best_estimator_") else est
    scores = cross_val_score(best, X_train, y_train, cv=cv, scoring="r2")
    fitted[name] = best
    comparison.append({
        "model": name,
        "cv_r2_mean": float(scores.mean()),
        "cv_r2_std": float(scores.std()),
    })
    print(f"{name:30s} CV R2 = {scores.mean():.3f} +/- {scores.std():.3f}")

best_name = max(comparison, key=lambda r: r["cv_r2_mean"])["model"]
model = fitted[best_name]
print(f"\nBest model by CV: {best_name}")

# ----------------------------------------------------------------------------
# 3. Honest evaluation on the untouched test set
# ----------------------------------------------------------------------------
pred_test = model.predict(X_test)
test_metrics = {
    "r2": float(r2_score(y_test, pred_test)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
    "mae": float(mean_absolute_error(y_test, pred_test)),
}
print("Test metrics:", {k: round(v, 3) for k, v in test_metrics.items()})

# ----------------------------------------------------------------------------
# 4. Prediction interval from out-of-fold residuals (80% interval)
# ----------------------------------------------------------------------------
oof = cross_val_predict(model, X_train, y_train, cv=5)
residuals = (y_train - oof).to_numpy()
interval = {
    "low": float(np.quantile(residuals, 0.10)),
    "high": float(np.quantile(residuals, 0.90)),
}

# ----------------------------------------------------------------------------
# 5. Feature importance + final refit on ALL data for deployment
# ----------------------------------------------------------------------------
if hasattr(model, "feature_importances_"):
    importance = dict(zip(FEATURES, map(float, model.feature_importances_)))
else:
    importance = dict(zip(FEATURES, map(float, np.abs(model.coef_) / np.abs(model.coef_).sum())))

test_points = pd.DataFrame({
    "actual": y_test.to_numpy(), "predicted": pred_test,
    "RM": X_test["RM"].to_numpy(),
}).round(0).to_dict(orient="list")

final_model = type(model)(**model.get_params()).fit(X, y)

joblib.dump(final_model, "model.joblib")
with open("metrics.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "comparison": comparison,
        "test": test_metrics,
        "interval": interval,
        "importance": importance,
        "test_points": test_points,
        "feature_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES},
    }, f, indent=2)
print("Saved model.joblib and metrics.json")
