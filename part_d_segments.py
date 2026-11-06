"""
Part D - Hub Segmentation

Reads:
    data/modelling_table.csv
    data/hubs.csv

Writes:
    outputs/hub_profile.csv
    outputs/part_d_separation.csv
    outputs/hub_segments.csv
    outputs/segment_profile.csv
    outputs/tier_vs_segment_mismatch.csv
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 2. Hub profiling
# --------------------------------------------------

required_columns = [
    "hub_id",
    "shipment_id",
    "delivery_hours",
    "is_delayed",
    "distance_km",
    "n_stops",
]

missing = [col for col in required_columns if col not in table.columns]

if missing:
    raise ValueError(f"Missing columns in modelling_table.csv: {missing}")

hub_profile = (
    table.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "size"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .reset_index()
    .sort_values("hub_id")
    .reset_index(drop=True)
)

if len(hub_profile) != 24:
    raise ValueError(
        f"Expected 24 hubs, found {len(hub_profile)}"
    )

if not (hub_profile["shipments"] == 50).all():
    raise ValueError("Every hub must have exactly 50 shipments.")

profile_columns = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]

if hub_profile[profile_columns].isna().any().any():
    raise ValueError("Hub profile contains missing values.")

hub_profile.to_csv(
    "outputs/hub_profile.csv",
    index=False,
    float_format="%.4f",
)


# --------------------------------------------------
# 3. Scale the four operational profile columns
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    hub_profile[profile_columns]
)


# --------------------------------------------------
# 4. K-Means experiments: K = 2, 3, 4, 5
# --------------------------------------------------

separation_rows = []
cluster_labels = {}

for k in [2, 3, 4, 5]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    cluster_labels[k] = labels

    # Calculate distances between all unique hub pairs.
    within_distances = []
    across_distances = []

    for i, j in combinations(range(len(X_scaled)), 2):

        distance = np.linalg.norm(
            X_scaled[i] - X_scaled[j]
        )

        if labels[i] == labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    if not within_distances:
        raise ValueError(
            f"No within-cluster hub pairs for K={k}"
        )

    if not across_distances:
        raise ValueError(
            f"No across-cluster hub pairs for K={k}"
        )

    mean_within = np.mean(within_distances)
    mean_across = np.mean(across_distances)

    separation_ratio = mean_across / mean_within

    cluster_sizes = sorted(
        np.bincount(labels).tolist(),
        reverse=True,
    )

    separation_rows.append({
        "K": k,
        "across_over_within": separation_ratio,
        "cluster_sizes": "-".join(
            str(size) for size in cluster_sizes
        ),
    })


separation = pd.DataFrame(separation_rows)

separation.to_csv(
    "outputs/part_d_separation.csv",
    index=False,
    float_format="%.4f",
)


# --------------------------------------------------
# 5. Final business segmentation using K = 3
# --------------------------------------------------

final_labels = cluster_labels[3]

profile_with_cluster = hub_profile.copy()
profile_with_cluster["cluster"] = final_labels

# Rank clusters by their mean hub-level delay rate.
cluster_delay_rates = (
    profile_with_cluster
    .groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

cluster_to_segment = {
    cluster_delay_rates.index[0]: "Standard",
    cluster_delay_rates.index[1]: "Watch",
    cluster_delay_rates.index[2]: "Escalate",
}

profile_with_cluster["segment"] = (
    profile_with_cluster["cluster"]
    .map(cluster_to_segment)
)


# --------------------------------------------------
# 6. Save hub_segments.csv
# --------------------------------------------------

hub_segments = (
    profile_with_cluster[["hub_id", "segment"]]
    .merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
        validate="one_to_one",
    )
    [["hub_id", "segment", "hub_city", "hub_tier"]]
    .sort_values("hub_id")
    .reset_index(drop=True)
)

if hub_segments.isna().any().any():
    raise ValueError("Missing values found in hub_segments.")

hub_segments.to_csv(
    "outputs/hub_segments.csv",
    index=False,
)


# --------------------------------------------------
# 7. Segment-level profile
# --------------------------------------------------

segment_profile = (
    profile_with_cluster
    .groupby("segment")
    .agg(
        hubs=("hub_id", "nunique"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
    .reset_index()
)

segment_order = {
    "Standard": 0,
    "Watch": 1,
    "Escalate": 2,
}

segment_profile["_order"] = (
    segment_profile["segment"].map(segment_order)
)

segment_profile = (
    segment_profile
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False,
    float_format="%.4f",
)


# --------------------------------------------------
# 8. Tier-versus-segment mismatches
# --------------------------------------------------

tier_to_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}

hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(tier_to_segment)
)

mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
].copy()

mismatches = mismatches[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].reset_index(drop=True)

mismatches.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False,
)


# --------------------------------------------------
# 9. Print results
# --------------------------------------------------

print("\n== HUB PROFILE ==")
print(hub_profile.to_string(index=False))

print("\n== K-MEANS SEPARATION ==")
print(separation.to_string(index=False))

print("\n== HUB SEGMENTS ==")
print(hub_segments.drop(columns="expected_segment").to_string(index=False))

print("\n== SEGMENT PROFILE ==")
print(segment_profile.to_string(index=False))

print("\n== TIER-VERSUS-SEGMENT MISMATCHES ==")
print(mismatches.to_string(index=False))

print("\nPart D segmentation completed.")