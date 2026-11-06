"""Part B - Delivery-hours regression.

Reads  : data/modelling_table.csv
Writes : outputs/part_b_checks.csv, part_b_tree_depth.csv, part_b_metrics.csv,
         part_b_gb_learning_rate.csv, part_b_cv.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["delivery_hours"]

# ---------------------------------------------------------------- task 1: split 70 / 15 / 15
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)

# ---------------------------------------------------------------- task 2: scaling without leakage
scaler = StandardScaler().fit(X_train)                  # fitted on the 840 training rows only
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)                  # test rows only transformed, never fitted on
pd.DataFrame([
    ("train_rows", len(X_train)),
    ("val_rows", len(X_val)),
    ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean", float(np.abs(train_scaled.mean(axis=0)).max())),   # kept unrounded (must be < 1e-9)
    ("test_scaled_max_abs_mean", round(float(np.abs(test_scaled.mean(axis=0)).max()), 4)),
], columns=["metric", "value"], dtype=object).to_csv("outputs/part_b_checks.csv", index=False)


def scores(model_name, split, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"model": model_name, "split": split,
            "MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------- task 3: baseline (mean of y_train)
baseline_pred = np.full(len(y_val), y_train.mean())
rows = [scores("mean_baseline", "val", y_val, baseline_pred)]

# ---------------------------------------------------------------- task 4: tree-depth diagnostic
depth_rows = []
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    depth_rows.append({
        "max_depth": "unlimited" if d is None else d,
        "train_rmse": rmse(y_train, tree.predict(X_train)),
        "val_rmse": rmse(y_val, tree.predict(X_val)),
        "_order": 10_000 if d is None else d,
    })
depth_df = pd.DataFrame(depth_rows)
# lowest val_rmse; ties -> smaller depth (unlimited counts as the largest depth)
best_row = depth_df.sort_values(["val_rmse", "_order"]).iloc[0]
best_depth = None if best_row["max_depth"] == "unlimited" else int(best_row["max_depth"])
depth_df.drop(columns="_order").round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

# ---------------------------------------------------------------- task 5: four candidate models
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    rows.append(scores(name, "val", y_val, model.predict(X_val)))
metrics = pd.DataFrame(rows)

# ---------------------------------------------------------------- task 6: learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    pred = gb.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": rmse(y_val, pred), "val_r2": r2_score(y_val, pred)})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# ---------------------------------------------------------------- task 7: cross-validation + final model
val_rank = metrics[metrics["model"] != "mean_baseline"].sort_values("RMSE")
top_two = list(val_rank["model"].iloc[:2])
final_name = top_two[0]                                  # lowest validation RMSE

X_trval = pd.concat([X_train, X_val])                    # the 1,020 training-plus-validation rows
y_trval = pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)    # ONE KFold object shared by both models
cv_rows = []
for name in top_two:
    r2s = cross_val_score(candidates[name], X_trval, y_trval, cv=kf)   # default scoring = R2
    cv_rows.append({"model": name, "mean_r2": r2s.mean(), "std_r2": r2s.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

# the final model is already fitted on the training rows; it touches the test rows exactly once
test_row = scores(final_name, "test", y_test, candidates[final_name].predict(X_test))
metrics = pd.concat([metrics, pd.DataFrame([test_row])], ignore_index=True)
metrics[["MAE", "MSE", "RMSE", "R2"]] = metrics[["MAE", "MSE", "RMSE", "R2"]].round(4)
metrics.to_csv("outputs/part_b_metrics.csv", index=False)

print("best tree depth:", best_depth, "| final model:", final_name)
for f in ["checks", "tree_depth", "metrics", "gb_learning_rate", "cv"]:
    print(f"\n== part_b_{f}.csv")
    print(pd.read_csv(f"outputs/part_b_{f}.csv").to_string(index=False))
