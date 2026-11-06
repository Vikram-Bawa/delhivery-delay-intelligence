"""Shared preprocessing for the Delhivery Last-Mile Delay Intelligence pipeline.

This is the ONLY place where cleaning and feature building are defined.
Part A, Part D's scoring script (predict_delay.py) and package_model.py all import from here,
so historical parcels and new parcels are always prepared in exactly the same way.
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no",
                 "sticker_serial_mod", "shelf_slot_no"]

# Columns that receive median imputation (keys of the returned medians dictionary).
MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]

RUPEE_SIGN = "\u20b9"
SENTINEL_WEIGHT_KG = 100  # the network accepts parcels up to 40 kg; anything above 100 is a scanner fault


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export.

    Steps, in this exact order:
      i.   drop exact duplicate rows
      ii.  service_type: strip spaces, title case (Express / Standard / Economy)
      iii. payment_mode: missing -> "Unknown"
      iv.  declared_value: remove rupee sign and commas, convert to a number,
           store in declared_value_inr, drop the text column
      v.   dispatch_time -> datetime
      vi.  parcel_weight_kg > 100 is a scanner-fault sentinel -> missing
      vii. fill missing distance_km, parcel_weight_kg, declared_value_inr with the column median.
           medians=None  -> compute them from the non-missing values left after steps i-vi
           medians=dict  -> reuse the supplied values (never recompute)

    Returns (clean_df, medians) where medians is a dict with float values.
    """
    df = raw.drop_duplicates().reset_index(drop=True)                        # i
    df["service_type"] = df["service_type"].str.strip().str.title()          # ii
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")                # iii
    df["declared_value_inr"] = pd.to_numeric(                                # iv
        df["declared_value"].str.replace(RUPEE_SIGN, "", regex=False).str.replace(",", "", regex=False)
    )
    df = df.drop(columns=["declared_value"])
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])                # v
    df.loc[df["parcel_weight_kg"] > SENTINEL_WEIGHT_KG, "parcel_weight_kg"] = np.nan   # vi

    if medians is None:                                                      # vii
        medians = {col: float(df[col].median()) for col in MEDIAN_COLUMNS}
    for col, value in medians.items():
        df[col] = df[col].fillna(value)
    return df, medians


def build_features(clean, hubs):
    """Attach hub columns (left merge on hub_id) and create the model-input columns.

    Never touches delivery_hours, so it works on parcels that have not been delivered yet.
    """
    out = clean.merge(
        hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
        on="hub_id", how="left",
    )
    out["promised_hours"] = out["service_type"].map(PROMISED_HOURS).astype(int)
    out["is_cod"] = (out["payment_mode"] == "COD").astype(int)
    out["is_weekend"] = (out["dispatch_time"].dt.day_of_week >= 5).astype(int)
    out["dispatch_hour"] = out["dispatch_time"].dt.hour.astype(int)
    out["log_declared_value"] = np.log(out["declared_value_inr"])
    return out
