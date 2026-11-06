"""
Phase 7 - New Shipment Scoring

Reads:
    data/new_shipments.csv
    data/hubs.csv
    models/delay_model.pkl
    models/delay_threshold.csv
    models/feature_order.csv
    models/impute_values.csv
    outputs/hub_segments.csv

Uses:
    src/shipment_prep.py

Writes:
    outputs/new_shipment_actions.csv
"""

from pathlib import Path

import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# =========================================================
# 1. PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. LOAD NEW SHIPMENTS AND HUB DATA
# =========================================================

new_shipments_path = DATA_DIR / "new_shipments.csv"
hubs_path = DATA_DIR / "hubs.csv"

new_shipments = pd.read_csv(new_shipments_path)
hubs = pd.read_csv(hubs_path)

print("New shipment data loaded.")
print(f"New shipments: {len(new_shipments)}")
print(f"Hub records: {len(hubs)}")


# =========================================================
# 3. LOAD HISTORICAL IMPUTATION MEDIANS
# =========================================================

impute_path = MODEL_DIR / "impute_values.csv"

impute_df = pd.read_csv(impute_path)

required_impute_columns = {"column", "median"}

if not required_impute_columns.issubset(impute_df.columns):
    raise ValueError(
        "impute_values.csv must contain 'column' and 'median' columns."
    )

if impute_df["column"].duplicated().any():
    raise ValueError("Duplicate columns found in impute_values.csv.")

historical_medians = dict(
    zip(
        impute_df["column"],
        impute_df["median"]
    )
)

required_medians = {
    "distance_km",
    "parcel_weight_kg",
    "declared_value_inr"
}

if set(historical_medians) != required_medians:
    raise ValueError(
        "Historical median columns do not match the required columns."
    )

print("\nHistorical medians loaded:")
print(historical_medians)


# =========================================================
# 4. PREPROCESS NEW SHIPMENTS USING SHARED FUNCTIONS
# =========================================================

# Reuse historical medians.
# Do not calculate new medians from the 48 shipments.

cleaned_shipments, returned_medians = clean_shipments(
    new_shipments,
    medians=historical_medians
)

# Confirm that historical medians were reused unchanged.

if returned_medians != historical_medians:
    raise ValueError(
        "Historical imputation medians were unexpectedly changed."
    )

# Reuse the shared feature-building function.

featured_shipments = build_features(
    cleaned_shipments,
    hubs
)

print("\nShared preprocessing completed.")


# =========================================================
# 5. LOAD THE EXACT FEATURE ORDER
# =========================================================

feature_order_path = MODEL_DIR / "feature_order.csv"

feature_order_df = pd.read_csv(feature_order_path)

if list(feature_order_df.columns) != ["feature"]:
    raise ValueError(
        "feature_order.csv must contain one column named 'feature'."
    )

feature_order = feature_order_df["feature"].tolist()

if len(feature_order) != 17:
    raise ValueError(
        f"Expected 17 features, found {len(feature_order)}."
    )

if len(set(feature_order)) != len(feature_order):
    raise ValueError("Duplicate feature names found.")

missing_features = [
    feature
    for feature in feature_order
    if feature not in featured_shipments.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features after preprocessing: {missing_features}"
    )

# Arrange features in the exact saved order.

X_new = featured_shipments[feature_order].copy()

if X_new.isna().any().any():
    raise ValueError(
        "Missing values remain in the new shipment feature matrix."
    )

print(f"\nFeature matrix prepared: {X_new.shape}")


# =========================================================
# 6. LOAD THE PACKAGED CLASSIFICATION MODEL
# =========================================================

model_path = MODEL_DIR / "delay_model.pkl"

model = joblib.load(model_path)

print("\nPackaged delay model loaded.")


# =========================================================
# 7. LOAD THE SELECTED ALERT THRESHOLD
# =========================================================

threshold_path = MODEL_DIR / "delay_threshold.csv"

threshold_df = pd.read_csv(threshold_path)

if list(threshold_df.columns) != ["threshold"]:
    raise ValueError(
        "delay_threshold.csv must contain one column named 'threshold'."
    )

if len(threshold_df) != 1:
    raise ValueError(
        "delay_threshold.csv must contain exactly one threshold."
    )

threshold = float(threshold_df.loc[0, "threshold"])

if not 0 <= threshold <= 1:
    raise ValueError("Threshold must be between 0 and 1.")

print(f"Selected alert threshold: {threshold}")


# =========================================================
# 8. CALCULATE DELAY PROBABILITIES
# =========================================================

delay_probabilities = model.predict_proba(X_new)[:, 1]

featured_shipments["delay_probability"] = delay_probabilities

# Flag shipments using the stored threshold.
# Compare the unrounded probabilities.

featured_shipments["flagged"] = (
    featured_shipments["delay_probability"] >= threshold
).astype(int)

print("\nDelay probabilities calculated.")

print(
    "Flagged shipments:",
    int(featured_shipments["flagged"].sum())
)


# =========================================================
# 9. LOAD HUB SEGMENT INFORMATION
# =========================================================

segments_path = OUTPUT_DIR / "hub_segments.csv"

hub_segments = pd.read_csv(segments_path)

required_segment_columns = {
    "hub_id",
    "segment"
}

if not required_segment_columns.issubset(hub_segments.columns):
    raise ValueError(
        "hub_segments.csv must contain hub_id and segment."
    )

if hub_segments["hub_id"].duplicated().any():
    raise ValueError(
        "Duplicate hub IDs found in hub_segments.csv."
    )

valid_segments = {
    "Standard",
    "Watch",
    "Escalate"
}

if not set(hub_segments["segment"].dropna()).issubset(valid_segments):
    raise ValueError("Unexpected hub segment label found.")


# =========================================================
# 10. MERGE HUB SEGMENTS WITH NEW SHIPMENTS
# =========================================================

featured_shipments = featured_shipments.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    validate="many_to_one"
)

if featured_shipments["segment"].isna().any():
    raise ValueError(
        "Some new shipments could not be matched to a hub segment."
    )

print("\nHub segments successfully merged.")


# =========================================================
# 11. ASSIGN OPERATIONAL ACTIONS
# =========================================================

def assign_action(row):
    """
    Apply the prescribed operational action rules.
    """

    if row["flagged"] == 0:
        return "No action"

    if row["segment"] == "Escalate":
        return "Escalate to hub manager"

    if row["segment"] in ["Watch", "Standard"]:
        return "Proactive customer call"

    raise ValueError(
        f"Unexpected segment: {row['segment']}"
    )


featured_shipments["action"] = featured_shipments.apply(
    assign_action,
    axis=1
)


# =========================================================
# 12. BUILD THE FINAL OUTPUT TABLE
# =========================================================

final_output = featured_shipments[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
].copy()


# =========================================================
# 13. ROUND PROBABILITIES TO FOUR DECIMAL PLACES
# =========================================================

final_output["delay_probability"] = (
    final_output["delay_probability"].round(4)
)


# =========================================================
# 14. ACCEPTANCE CHECKS
# =========================================================

# Exactly 48 shipments.

assert len(final_output) == 48, (
    f"Expected 48 shipments, found {len(final_output)}."
)

# Shipment IDs must be unique.

assert final_output["shipment_id"].is_unique, (
    "Duplicate shipment IDs detected."
)

# No missing values.

assert not final_output.isna().any().any(), (
    "Missing values found in the final output."
)

# Probability values must be within the valid range.

assert final_output["delay_probability"].between(0, 1).all()

# Flags must be binary.

assert set(final_output["flagged"].unique()).issubset({0, 1})

# Verify the operational action rules.

for _, row in final_output.iterrows():

    if row["flagged"] == 0:

        assert row["action"] == "No action"

    elif row["segment"] == "Escalate":

        assert row["action"] == "Escalate to hub manager"

    elif row["segment"] in ["Watch", "Standard"]:

        assert row["action"] == "Proactive customer call"


# =========================================================
# 15. SAVE THE FINAL OPERATIONAL OUTPUT
# =========================================================

output_path = OUTPUT_DIR / "new_shipment_actions.csv"

final_output.to_csv(
    output_path,
    index=False,
    float_format="%.4f"
)

print("\nFinal output saved:")
print(output_path)


# =========================================================
# 16. DISPLAY RESULTS
# =========================================================

print("\nFinal shipment actions:")
print(final_output.to_string(index=False))

print("\nOperational summary:")

print(
    "Total shipments:",
    len(final_output)
)

print(
    "Flagged shipments:",
    int(final_output["flagged"].sum())
)

print(
    "Unflagged shipments:",
    int((final_output["flagged"] == 0).sum())
)

print("\nAction counts:")

print(
    final_output["action"].value_counts().to_string()
)

print("\nSegment counts:")

print(
    final_output["segment"].value_counts().to_string()
)


# =========================================================
# 17. FINAL COMPLETION MESSAGE
# =========================================================

print("\n" + "=" * 60)
print("PHASE 7: NEW SHIPMENT SCORING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nAll acceptance checks passed.")

print("\nGenerated file:")
print("outputs/new_shipment_actions.csv")