"""
Phase 6 - Model Packaging

Reads:
    data/modelling_table.csv
    data/raw_shipments.csv
    outputs/part_c_final_config.csv
    src/shipment_prep.py

Writes:
    models/delay_model.pkl
    models/delay_threshold.csv
    models/feature_order.csv
    models/impute_values.csv
    outputs/part_d_reload_check.csv
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


# =========================================================
# 1. PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. LOAD MODELLING DATA
# =========================================================

table = pd.read_csv(DATA_DIR / "modelling_table.csv")

X = table[FEATURES].copy()
y = table["is_delayed"].astype(int)

print("Modelling data loaded.")
print(f"Total shipments: {len(table)}")
print(f"Total features: {len(FEATURES)}")


# =========================================================
# 3. RECREATE THE EXACT PART C DATA SPLIT
# =========================================================

# First split:
# 70% training = 840 rows
# 30% remaining = 360 rows

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split:
# Divide the remaining 360 rows equally.
# Validation = 180 rows
# Test = 180 rows

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Verify split sizes.

assert len(X_train) == 840, "Incorrect training split."
assert len(X_val) == 180, "Incorrect validation split."
assert len(X_test) == 180, "Incorrect test split."

# Verify delayed shipment counts.

assert int(y_train.sum()) == 195, "Incorrect training delayed count."
assert int(y_val.sum()) == 42, "Incorrect validation delayed count."
assert int(y_test.sum()) == 41, "Incorrect test delayed count."

print("\nData split verified:")
print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows:       {len(X_test)}")

print("\nDelayed shipment counts:")
print(f"Training:   {int(y_train.sum())}")
print(f"Validation: {int(y_val.sum())}")
print(f"Test:       {int(y_test.sum())}")


# =========================================================
# 4. LOAD FINAL PART C CONFIGURATION
# =========================================================

config_path = OUTPUT_DIR / "part_c_final_config.csv"

config_df = pd.read_csv(config_path)

if len(config_df) != 1:
    raise ValueError(
        "part_c_final_config.csv must contain exactly one row."
    )

config = config_df.iloc[0]


def get_config_value(possible_names):
    """Retrieve a configuration value using accepted column names."""

    for name in possible_names:
        if name in config_df.columns:
            return config[name]

    raise KeyError(
        f"Could not find configuration column from {possible_names}. "
        f"Available columns: {list(config_df.columns)}"
    )


selector_k = int(
    get_config_value(["selector_k", "k"])
)

n_estimators = int(
    get_config_value(["n_estimators"])
)

max_depth_value = get_config_value(["max_depth"])

if (
    pd.isna(max_depth_value)
    or str(max_depth_value).strip().lower()
    in ["unlimited", "none", "nan", ""]
):
    max_depth = None
else:
    max_depth = int(max_depth_value)

min_samples_leaf = int(
    get_config_value(["min_samples_leaf"])
)

threshold = float(
    get_config_value(["threshold", "selected_threshold"])
)

print("\nFinal Part C configuration:")
print(f"Selector K:       {selector_k}")
print(f"Estimators:       {n_estimators}")
print(f"Maximum depth:    {max_depth}")
print(f"Minimum leaf:     {min_samples_leaf}")
print(f"Threshold:        {threshold}")


# =========================================================
# 5. VERIFY THE SELECTED CONFIGURATION
# =========================================================

assert selector_k == 7, "Expected selector_k = 7."

assert n_estimators == 100, "Expected n_estimators = 100."

assert max_depth is None, "Expected unlimited max_depth."

assert min_samples_leaf == 1, (
    "Expected min_samples_leaf = 1."
)

assert threshold == 0.5, "Expected threshold = 0.5."


# =========================================================
# 6. REBUILD THE FINAL PART C PIPELINE
# =========================================================

pipeline = Pipeline([
    (
        "select",
        SelectKBest(
            score_func=f_classif,
            k=selector_k
        )
    ),

    (
        "classifier",
        RandomForestClassifier(
            random_state=42,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
    )
])


# Fit ONLY on the original training data.

pipeline.fit(X_train, y_train)

print("\nFinal classification pipeline fitted.")
print("Only the 840 training rows were used.")


# =========================================================
# 7. SAVE THE FITTED MODEL USING JOBLIB
# =========================================================

model_path = MODEL_DIR / "delay_model.pkl"

joblib.dump(pipeline, model_path)

print(f"\nModel saved: {model_path}")


# =========================================================
# 8. SAVE THE SELECTED THRESHOLD
# =========================================================

threshold_df = pd.DataFrame({
    "threshold": [threshold]
})

threshold_df.to_csv(
    MODEL_DIR / "delay_threshold.csv",
    index=False,
    float_format="%.4f"
)

print("Threshold saved.")


# =========================================================
# 9. SAVE THE EXACT FEATURE ORDER
# =========================================================

feature_df = pd.DataFrame({
    "feature": FEATURES
})

feature_df.to_csv(
    MODEL_DIR / "feature_order.csv",
    index=False
)

print(f"Feature order saved: {len(FEATURES)} features.")


# =========================================================
# 10. SAVE HISTORICAL IMPUTATION MEDIANS
# =========================================================

raw_path = DATA_DIR / "raw_shipments.csv"

raw_shipments = pd.read_csv(raw_path)

cleaned_result = clean_shipments(raw_shipments)

if not isinstance(cleaned_result, tuple) or len(cleaned_result) != 2:
    raise TypeError(
        "clean_shipments() must return "
        "(cleaned_dataframe, historical_medians)."
    )

cleaned_shipments, historical_medians = cleaned_result

if not isinstance(historical_medians, dict):
    raise TypeError(
        "Historical imputation medians must be returned as a dictionary."
    )

if not historical_medians:
    raise ValueError(
        "No historical imputation medians were returned."
    )

impute_df = pd.DataFrame([
    {
        "column": column,
        "median": value
    }
    for column, value in historical_medians.items()
])

impute_df.to_csv(
    MODEL_DIR / "impute_values.csv",
    index=False,
    float_format="%.4f"
)

print("\nHistorical imputation medians saved:")
print(impute_df.to_string(index=False))


# =========================================================
# 11. RELOAD THE SAVED MODEL
# =========================================================

reloaded_pipeline = joblib.load(model_path)

print("\nSaved model successfully reloaded.")


# =========================================================
# 12. COMPARE ORIGINAL AND RELOADED MODEL PROBABILITIES
# =========================================================

original_probabilities = pipeline.predict_proba(X_test)[:, 1]

reloaded_probabilities = (
    reloaded_pipeline.predict_proba(X_test)[:, 1]
)

max_difference = float(
    abs(
        original_probabilities - reloaded_probabilities
    ).max()
)

print(
    "\nMaximum absolute probability difference:",
    max_difference
)


# =========================================================
# 13. SAVE RELOAD VERIFICATION RESULTS
# =========================================================

reload_check = pd.DataFrame({
    "metric": [
        "test_rows_scored",
        "reload_max_abs_probability_diff",
        "threshold"
    ],

    "value": [
        len(X_test),
        max_difference,
        threshold
    ]
})

reload_check.to_csv(
    OUTPUT_DIR / "part_d_reload_check.csv",
    index=False,
    float_format="%.4f"
)

print("\nReload verification results:")
print(reload_check.to_string(index=False))


# =========================================================
# 14. FINAL ACCEPTANCE CHECKS
# =========================================================

assert max_difference < 1e-12, (
    "Reloaded model probabilities do not match "
    "the original model probabilities."
)

assert len(feature_df) == 17, (
    "Expected exactly 17 features."
)

assert len(impute_df) > 0, (
    "Imputation values are missing."
)

assert model_path.exists()

assert (MODEL_DIR / "delay_threshold.csv").exists()

assert (MODEL_DIR / "feature_order.csv").exists()

assert (MODEL_DIR / "impute_values.csv").exists()

assert (OUTPUT_DIR / "part_d_reload_check.csv").exists()


# =========================================================
# 15. COMPLETION MESSAGE
# =========================================================

print("\n" + "=" * 55)
print("PHASE 6: MODEL PACKAGING COMPLETED SUCCESSFULLY")
print("=" * 55)

print("\nGenerated files:")

print("1. models/delay_model.pkl")
print("2. models/delay_threshold.csv")
print("3. models/feature_order.csv")
print("4. models/impute_values.csv")
print("5. outputs/part_d_reload_check.csv")

print("\nAll acceptance checks passed.")