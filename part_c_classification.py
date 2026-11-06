"""Part C - Delay classifier and alert threshold.

Reads  : data/modelling_table.csv
Writes : outputs/part_c_models.csv, part_c_feature_scores.csv, part_c_selectk.csv,
         part_c_tuning.csv, part_c_thresholds.csv, part_c_final_test.csv, part_c_final_config.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]


def cell_metrics(y_true, y_pred):
    """Confusion-matrix cells plus accuracy, precision, recall, F1 (precision is 0 when nothing is predicted)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def write(df, path):
    """Every decimal rounded to 4 places, index=False."""
    df.round(4).to_csv(path, index=False)


# ---------------------------------------------------------------- task 1: stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
assert (int(y_train.sum()), int(y_val.sum()), int(y_test.sum())) == (195, 42, 41)

# ---------------------------------------------------------------- task 2: blind baseline + three classifiers
rows = [{"model": "blind_all_on_time", **cell_metrics(y_val, np.zeros(len(y_val), dtype=int))}]

logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
rows.append({"model": "logistic_regression", **cell_metrics(y_val, logit.predict(X_val))})

tree_scores = {}
for d in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    tree_scores[d] = f1_score(y_val, tree.predict(X_val))
best_d = max(tree_scores, key=lambda d: (tree_scores[d], -d))        # highest F1, ties -> smaller depth
best_tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
rows.append({"model": f"decision_tree_depth_{best_d}", **cell_metrics(y_val, best_tree.predict(X_val))})

forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append({"model": "random_forest_default", **cell_metrics(y_val, forest.predict(X_val))})
write(pd.DataFrame(rows), "outputs/part_c_models.csv")

# ---------------------------------------------------------------- task 3: which columns carry signal?
selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector_all.scores_,
    "rf_importance": forest.feature_importances_,
})
scores["f_rank"] = scores["f_score"].rank(ascending=False, method="first").astype(int)   # 1 = highest F
scores["is_noise"] = scores["feature"].isin(NOISE_COLUMNS)
scores = scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
# Combined importance share of the five noise columns
noise_share = (
    scores.loc[scores["is_noise"], "rf_importance"].sum()
    / scores["rf_importance"].sum()
)

print("\n== Feature importance diagnostics ==")
print("Top 7 features:")
print(scores.head(7)[["feature", "f_score", "f_rank"]].to_string(index=False))

print(f"\nCombined noise-column importance share: {noise_share:.4f}")

print("\nNoise columns:")
print(
    scores[scores["is_noise"]][
        ["feature", "f_score", "f_rank", "rf_importance"]
    ].to_string(index=False)
)
write(scores, "outputs/part_c_feature_scores.csv")

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val))})
write(pd.DataFrame(selectk_rows), "outputs/part_c_selectk.csv")

# ---------------------------------------------------------------- task 4: tuning (from here on k = 7)
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)          # fitted on training rows only
Xtr7, Xva7 = selector.transform(X_train), selector.transform(X_val)   # the test rows are not touched here

param_grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5).fit(Xtr7, y_train)
rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid,
                          n_iter=9, cv=5, random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)


def clean_params(p):
    return {key: int(v) for key, v in p.items()}


tuning = []
chosen_params = {}
for name, search in [("grid_search", grid), ("randomized_search", rand)]:
    n_cand = len(search.cv_results_["params"])
    best = clean_params(search.best_params_)
    chosen_params[name] = best
    tuning.append({
        "search": name, "candidates": n_cand, "cv_folds": 5, "total_fits": n_cand * 5,
        "best_params": str(best), "best_cv_accuracy": search.best_score_,
        "val_f1": f1_score(y_val, search.best_estimator_.predict(Xva7)),
    })
chosen_params["default_forest_same_7_features"] = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
tuning.append({
    "search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
    "best_params": "defaults", "best_cv_accuracy": np.nan,
    "val_f1": f1_score(y_val, default7.predict(Xva7)),
})
tuning_df = pd.DataFrame(tuning)
write(tuning_df, "outputs/part_c_tuning.csv")

# ---------------------------------------------------------------- task 5: choose the final configuration
tie_order = ["default_forest_same_7_features", "grid_search", "randomized_search"]
chosen_from = max(tie_order, key=lambda s: (tuning_df.set_index("search").loc[s, "val_f1"], -tie_order.index(s)))
params = chosen_params[chosen_from]
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=K)),
    ("model", RandomForestClassifier(random_state=42, **params)),
]).fit(X_train, y_train)

# ---------------------------------------------------------------- task 6: alert threshold
val_proba = final_model.predict_proba(X_val)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]            # exact decimals, no np.arange
thr_rows = []
for t in thresholds:
    pred = (val_proba >= t).astype(int)
    thr_rows.append({"threshold": t, **cell_metrics(y_val, pred)})
thr_df = pd.DataFrame(thr_rows)
write(thr_df, "outputs/part_c_thresholds.csv")

# Operations rule: a missed delayed parcel costs more than an unnecessary call ->
# the HIGHEST threshold whose validation recall is at least 0.80 (none reaches it -> 0.1).
ok = [r["threshold"] for r in thr_rows if r["recall"] >= 0.80]
threshold = max(ok) if ok else 0.1

# ---------------------------------------------------------------- task 7: one test evaluation
test_pred = (final_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
write(pd.DataFrame([{"threshold": threshold, **cell_metrics(y_test, test_pred)}]), "outputs/part_c_final_test.csv")

write(pd.DataFrame([{
    "selector_k": K,
    "n_estimators": params["n_estimators"],
    "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
    "min_samples_leaf": params["min_samples_leaf"],
    "threshold": threshold,
    "chosen_from": chosen_from,
}]), "outputs/part_c_final_config.csv")

for f in ["models", "feature_scores", "selectk", "tuning", "thresholds", "final_test", "final_config"]:
    print(f"\n== part_c_{f}.csv")
    print(pd.read_csv(f"outputs/part_c_{f}.csv").to_string(index=False))
