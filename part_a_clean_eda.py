"""Part A - Data foundation and delay evidence.

Reads  : data/raw_shipments.csv, data/hubs.csv
Writes : data/modelling_table.csv
         outputs/part_a_checks.csv, outputs/part_a_delay_by_group.csv, outputs/part_a_correlations.csv
"""
import os

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, clean_shipments, build_features

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# ---------------------------------------------------------------- clean + features + target
clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)

# Keep every column. Identifiers first, then the 17 features in pinned order, then the target columns.
lead = ["shipment_id", "hub_id"] + FEATURES + ["delivery_hours", "is_delayed"]
rest = [c for c in table.columns if c not in lead]
table = table[lead + rest]
table.to_csv("data/modelling_table.csv", index=False)

# ---------------------------------------------------------------- data-quality checks
# "Before" counts are measured after removing duplicates and before any filling.
dedup = raw.drop_duplicates()
hours = table["delivery_hours"]
q1, q3 = hours.quantile(0.25), hours.quantile(0.75)        # unrounded quartiles
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr              # bounds from unrounded quartiles

checks = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", int(dedup["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(dedup["parcel_weight_kg"].isna().sum())),   # sentinels (999) are not NaN
    ("sentinel_weight_rows", int((dedup["parcel_weight_kg"] > 100).sum())),
    ("missing_declared_value_before", int(dedup["declared_value"].isna().sum())),
    ("missing_payment_mode_before", int(dedup["payment_mode"].isna().sum())),
    ("service_type_spellings_before", int(dedup["service_type"].nunique())),
    ("service_type_spellings_after", int(table["service_type"].nunique())),
    # the four declared-value metrics use the median-FILLED column
    ("declared_value_mean_inr", table["declared_value_inr"].mean()),
    ("declared_value_median_inr", table["declared_value_inr"].median()),
    ("log_declared_value_mean", table["log_declared_value"].mean()),
    ("log_declared_value_median", table["log_declared_value"].median()),
    # the historical medians that fill the gaps (and are stored for the scoring script)
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(table)),
    ("nulls_in_features_and_target", int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(table["is_delayed"].sum())),
    ("delay_rate", table["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower),
    ("hours_upper_bound", upper),
    ("hours_outlier_count", int(((hours < lower) | (hours > upper)).sum())),
]
# Round only when writing; integers stay integers.
checks = [(m, v if isinstance(v, int) else round(float(v), 4)) for m, v in checks]
checks_df = pd.DataFrame(checks, columns=["metric", "value"], dtype=object)   # object keeps ints as ints
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

# ---------------------------------------------------------------- delay evidence by group
table["distance_band"] = pd.cut(
    table["distance_km"], bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"], right=True,
)
frames = []
for dimension in ["service_type", "hub_tier", "distance_band"]:
    g = table.groupby(dimension, observed=True).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index().rename(columns={dimension: "group"})
    g.insert(0, "dimension", dimension)
    g["group"] = g["group"].astype(str)
    frames.append(g)
by_group = pd.concat(frames, ignore_index=True)
by_group[["delay_rate", "mean_hours"]] = by_group[["delay_rate", "mean_hours"]].round(4)
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# ---------------------------------------------------------------- correlations (numeric columns, row labels kept)
corr = table.select_dtypes(include="number").corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv")

print("modelling table:", table.shape)
print(checks_df.to_string(index=False))
print(by_group.to_string(index=False))
