#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import joblib


# -------------------------
# Config
# -------------------------

BASE_DIR = Path("data/nhanes")

CYCLES = ["2015-2016", "2017-2020"]

ID_COL = "SEQN"

# Columns you definitely do NOT want to feed to TabDDPM
# (IDs, survey weights, etc. – adjust as needed)
EXCLUDE_PREFIXES = [
    "WT",   # weights
    "SDMV", # survey design vars
]
EXCLUDE_EXACT = [
    ID_COL,
]

# Columns you know are continuous and should NOT be treated as categorical
# even if they have few unique values in your subset
FORCE_CONTINUOUS = [
    "RIDAGEYR",  # age in years at screening
    # add more here if needed
]

# NHANES special missing codes: 7/77/777 etc. often mean "Refused", 9/99/999 =
# "Don't know". This is a generic list; tweak for your chosen variables.
SPECIAL_MISSING_CODES = {
    7, 9, 77, 99, 777, 999, 7777, 9999
}

# Thresholds
MAX_MISSING_COL = 0.5   # drop columns with > 50% missing
MAX_MISSING_ROW = 0.5   # drop rows with > 50% missing


# -------------------------
# Helpers
# -------------------------

def load_cycle(cycle_name: str) -> pd.DataFrame:
    """Load and merge all .xpt files in one cycle folder by SEQN."""
    cycle_dir = BASE_DIR / cycle_name
    xpt_files = sorted(cycle_dir.glob("*.XPT")) + sorted(cycle_dir.glob("*.xpt"))
    if not xpt_files:
        raise FileNotFoundError(f"No .xpt files found in {cycle_dir}")

    dfs = []
    for f in xpt_files:
        print(f"Reading {f} ...")
        df = pd.read_sas(f, format="xport")
        # Ensure SEQN is present
        if ID_COL not in df.columns:
            print(f"Warning: {f} has no {ID_COL}, skipping.")
            continue
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid .xpt files with {ID_COL} in {cycle_dir}")

    # Outer merge all on SEQN
    merged = reduce(
        lambda left, right: pd.merge(left, right, on=ID_COL, how="outer"),
        dfs
    )
    merged["cycle"] = cycle_name
    return merged


def apply_special_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Convert NHANES special codes (7,9,77,...) to NaN for numeric-like columns."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mask = df[col].isin(SPECIAL_MISSING_CODES)
            if mask.any():
                df.loc[mask, col] = np.nan
    return df


def drop_sparse(df: pd.DataFrame) -> pd.DataFrame:
    """Drop very sparse columns and rows."""
    df = df.copy()
    # drop columns
    col_missing = df.isna().mean()
    keep_cols = col_missing[col_missing <= MAX_MISSING_COL].index
    dropped_cols = set(df.columns) - set(keep_cols)
    if dropped_cols:
        print(f"Dropping {len(dropped_cols)} cols with > {MAX_MISSING_COL*100:.0f}% missing")

    df = df[keep_cols]

    # drop rows
    row_missing = df.isna().mean(axis=1)
    keep_rows = row_missing <= MAX_MISSING_ROW
    dropped_rows = (~keep_rows).sum()
    if dropped_rows:
        print(f"Dropping {dropped_rows} rows with > {MAX_MISSING_ROW*100:.0f}% missing")

    df = df.loc[keep_rows].reset_index(drop=True)
    return df


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID and other obviously non-feature columns."""
    cols = []
    for c in df.columns:
        if c in EXCLUDE_EXACT:
            continue
        if any(c.startswith(pref) for pref in EXCLUDE_PREFIXES):
            continue
        cols.append(c)
    dropped = set(df.columns) - set(cols)
    if dropped:
        print(f"Dropping {len(dropped)} non-feature columns: {sorted(dropped)[:10]} ...")
    return df[cols]


def infer_column_types(df: pd.DataFrame):
    """
    Infer categorical vs continuous columns.

    Heuristic:
      - Non-numeric -> categorical
      - Numeric:
          * If in FORCE_CONTINUOUS -> continuous
          * Else if n_unique <= 20 -> categorical
          * Else -> continuous
    """
    categorical_cols = []
    continuous_cols = []

    for col in df.columns:
        if col == "cycle":
            # treat cycle as categorical
            categorical_cols.append(col)
            continue

        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            if col in FORCE_CONTINUOUS:
                continuous_cols.append(col)
            else:
                n_unique = series.nunique(dropna=True)
                if n_unique <= 20:
                    categorical_cols.append(col)
                else:
                    continuous_cols.append(col)
        else:
            categorical_cols.append(col)

    print(f"Inferred {len(continuous_cols)} continuous and {len(categorical_cols)} categorical columns.")
    return continuous_cols, categorical_cols


def encode_for_tabddpm(df: pd.DataFrame,
                       continuous_cols,
                       categorical_cols,
                       save_prefix: Path):
    """
    Encode df into a fully numeric DataFrame for TabDDPM:
      - Continuous cols: StandardScaler
      - Categorical cols: OrdinalEncoder

    Returns encoded_df and metadata dict.
    """
    df = df.copy()

    # Continuous
    scaler = StandardScaler()
    if continuous_cols:
        df_cont = scaler.fit_transform(df[continuous_cols])
        df_cont = pd.DataFrame(df_cont, columns=continuous_cols, index=df.index)
    else:
        df_cont = pd.DataFrame(index=df.index)

    # Categorical
    if categorical_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df_cat_vals = enc.fit_transform(df[categorical_cols].astype("category"))
        df_cat = pd.DataFrame(df_cat_vals, columns=categorical_cols, index=df.index)
    else:
        enc = None
        df_cat = pd.DataFrame(index=df.index)

    # Combine back
    encoded_df = pd.concat([df_cont, df_cat], axis=1)

    # Build metadata
    metadata = {
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
        "encoded_columns": list(encoded_df.columns),
        "scaler": "standard_scaler.pkl",
        "encoder": "ordinal_encoder.pkl",
        "note": "categorical columns are ordinal-encoded; provide their indices to TabDDPM as discrete_columns."
    }

    # Save preprocessing objects
    joblib.dump(scaler, save_prefix.with_name(save_prefix.stem + "_standard_scaler.pkl"))
    if enc is not None:
        joblib.dump(enc, save_prefix.with_name(save_prefix.stem + "_ordinal_encoder.pkl"))

    return encoded_df, metadata


# -------------------------
# Main
# -------------------------

def main(args):
    # 1. Load and merge cycles
    all_cycles = []
    for c in CYCLES:
        print(f"=== Loading cycle {c} ===")
        df_c = load_cycle(c)
        all_cycles.append(df_c)

    df = pd.concat(all_cycles, axis=0, ignore_index=True)

    print(f"Combined raw shape: {df.shape}")

    # 2. Apply NHANES missing-code cleaning
    df = apply_special_missing(df)

    # 3. Drop columns/rows with too much missingness
    df = drop_sparse(df)

    # 4. Drop IDs & non-feature columns
    df_features = drop_unwanted_columns(df)

    print(f"Shape after dropping non-feature cols: {df_features.shape}")

    # 5. OPTIONAL: If you have a known label/target, keep it separately
    # Example:
    LABEL_COL = "SLQ040"  # 0/1 label you derive from questionnaire
    y = df_features[LABEL_COL] 
    df_features = df_features.drop(columns=[LABEL_COL])

    # 6. Infer col types
    continuous_cols, categorical_cols = infer_column_types(df_features)

    # 7. Encode for TabDDPM
    save_path = Path(args.output)
    encoded_df, metadata = encode_for_tabddpm(df_features, continuous_cols,
                                              categorical_cols, save_prefix=save_path)

    # 8. Save results
    print(f"Saving processed table to {save_path}")
    encoded_df.to_parquet(save_path)

    meta_path = save_path.with_suffix(".metadata.json")
    print(f"Saving metadata to {meta_path}")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NHANES XPT files for TabDDPM.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/nhanes_tabddpm.parquet",
        help="Path to save the processed TabDDPM-ready table."
    )
    args = parser.parse_args()
    main(args)
