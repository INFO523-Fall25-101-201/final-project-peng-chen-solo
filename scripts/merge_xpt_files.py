#!/usr/bin/env python
"""
Memory-efficient NHANES merger.

- Incrementally merge all .xpt files in 2015-2016 and 2017-2020 by SEQN.
- Store intermediate merged results as Parquet on disk (per cycle).
- Optionally combine per-cycle tables into one big table (can still be heavy).
- Create a subset table with selected columns and enforced dtypes
  (continuous vs categorical) and save it to disk.

No data cleaning or imputation is performed here.
"""

from pathlib import Path
from typing import List
import pandas as pd
import gc

# ----------------------------------------------------------------------
# CONFIGURATION – EDIT THESE
# ----------------------------------------------------------------------

BASE_DIR = Path("data/nhanes")
CYCLES = ["2015-2016", "2017-2020"]

OUTPUT_DIR = Path("data/processed")

# Per-cycle merged tables (full width)
CYCLE_PARQUET_TEMPLATE = OUTPUT_DIR / "nhanes_{cycle}_merged.parquet"

# Optionally build a single big full table from both cycles
BUILD_FULL_MERGED_TABLE = False  # set to True if you really want it

FULL_MERGED_PARQUET = OUTPUT_DIR / "nhanes_2015_2020_merged.parquet"

# Subset outputs (this is what you’ll likely feed to TabDDPM)
SUBSET_PARQUET = OUTPUT_DIR / "nhanes_2015_2020_subset.parquet"
SUBSET_CSV = OUTPUT_DIR / "nhanes_2015_2020_subset.csv"

# ---- Column subsets & dtypes (FILL THESE) ----------------------------
CONTINUOUS_COLS: List[str] = [
    # e.g. "RIDAGEYR", "BMXBMI", "LBXTC", ...
    "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "RIDAGEYR", "INDFMPIR", "SLD012", "PAD630", "PAD680", "BMXWAIST", "LBXSLDSI", "LBXPLTSI ", "LBXSCR", "LBXHSCRP", "BPXSY3", "BPXDI3", "BMXBMI", "LBXTC", "LBDHDD", "LBXGH", "LBXSAL"
]

CATEGORICAL_COLS: List[str] = [
    # e.g. "RIAGENDR", "RIDRETH1", "SMQ020", ...
    "RIAGENDR", "RIDRETH1", "DMDMARTL", "DMDEDU2", "SLQ030", "SLQ040", "SLQ050", "SLQ120", "SMQ040", "ALQ110 ", "MCQ220"
]

# Extra columns you always want to keep
EXTRA_COLUMNS: List[str] = [
    "SEQN",
    "cycle",
]


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def load_xpt_file(path: Path) -> pd.DataFrame:
    """
    Load a single XPT file using pandas.read_sas.
    Returns an empty DataFrame if SEQN is not present.
    """
    print(f"  Loading {path}")
    df = pd.read_sas(path, format="xport", encoding="latin1")

    if "SEQN" not in df.columns:
        print(f"    Warning: 'SEQN' not found in {path.name}, skipping.")
        return pd.DataFrame()

    # Drop duplicate columns if any inside this file
    df = df.loc[:, ~df.columns.duplicated()]

    # Some files may have repeated SEQN rows; collapse to one row per SEQN
    if df["SEQN"].duplicated().any():
        df = df.sort_values("SEQN").drop_duplicates("SEQN", keep="first")

    return df


def merge_cycle_incremental(cycle_name: str) -> Path:
    """
    For a given cycle (e.g. '2015-2016'):

    - Iterate through XPT files.
    - For the first file: write it to a Parquet file as the base.
    - For each subsequent file:
      * Load current merged Parquet
      * Drop from the new XPT any columns (except SEQN) that already
        exist in the merged table, with a warning.
      * Outer-merge with new XPT table by SEQN
      * Write merged result back to the same Parquet path

    This avoids suffix collisions and keeps memory usage moderate.
    """
    cycle_dir = BASE_DIR / cycle_name
    if not cycle_dir.exists():
        raise FileNotFoundError(f"Cycle directory not found: {cycle_dir}")

    xpt_files = sorted(cycle_dir.glob("*.[Xx][Pp][Tt]"))
    if not xpt_files:
        raise FileNotFoundError(f"No .xpt files found in {cycle_dir}")

    out_path = CYCLE_PARQUET_TEMPLATE.with_name(
        CYCLE_PARQUET_TEMPLATE.name.format(cycle=cycle_name.replace("-", "_"))
    )
    print(f"\n=== Processing cycle {cycle_name} ===")
    print(f"Found {len(xpt_files)} XPT files in {cycle_dir}")

    merged_path_exists = False

    for idx, xpt in enumerate(xpt_files):
        df_new = load_xpt_file(xpt)
        if df_new.empty:
            continue

        if not merged_path_exists:
            # First valid file for this cycle
            df_new["cycle"] = cycle_name
            print(f"  Initializing merged table for {cycle_name} with {xpt.name}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_new.to_parquet(out_path, index=False)
            merged_path_exists = True
            del df_new
            gc.collect()
        else:
            # Incrementally merge with existing cycle parquet
            print(f"  Merging {xpt.name} into existing {out_path.name}")
            df_merged = pd.read_parquet(out_path)

            # --------- NEW LOGIC: DROP DUPLICATE COLUMNS FROM df_new ---------
            existing_cols = set(df_merged.columns)
            dup_cols = [
                c for c in df_new.columns
                if c != "SEQN" and c in existing_cols
            ]
            if dup_cols:
                for col in dup_cols:
                    print(
                        f"    Warning: duplicate column '{col}' in file {xpt.name} "
                        f"for cycle {cycle_name} – dropping this column from new file."
                    )
                df_new = df_new.drop(columns=dup_cols)
            # -----------------------------------------------------------------

            # Outer join on SEQN; no suffixes needed now
            df_merged = df_merged.merge(df_new, on="SEQN", how="outer")

            # Write back to the same Parquet file
            df_merged.to_parquet(out_path, index=False)

            # Clean up
            del df_new, df_merged
            gc.collect()

    if not merged_path_exists:
        raise ValueError(f"No valid XPT files with SEQN for cycle {cycle_name}")

    # Quick report
    df_check = pd.read_parquet(out_path, columns=["SEQN"])
    print(
        f"Finished cycle {cycle_name}: {len(df_check)} rows (unique SEQN). "
        f"Saved to {out_path}"
    )
    del df_check
    gc.collect()

    return out_path


def enforce_dtypes(df: pd.DataFrame,
                   continuous_cols: List[str],
                   categorical_cols: List[str]) -> pd.DataFrame:
    """
    Enforce:
      - continuous columns -> numeric
      - categorical columns -> 'category'
    Nonexistent columns are silently ignored.
    """
    df = df.copy()

    for col in continuous_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def build_full_merged_table(cycle_paths: List[Path]) -> None:
    """
    Combine per-cycle merged Parquet files into one big table.

    NOTE: This *will* load both cycles into memory simultaneously,
    so it can still be heavy. Use only if you really need a single file.
    """
    print("\n=== Building full merged table across all cycles ===")
    dfs = []
    for p in cycle_paths:
        print(f"  Loading {p}")
        dfs.append(pd.read_parquet(p))

    full_df = pd.concat(dfs, axis=0, ignore_index=True)
    print(
        f"Full merged table shape: {full_df.shape[0]} rows, {full_df.shape[1]} columns"
    )
    full_df.to_parquet(FULL_MERGED_PARQUET, index=False)
    print(f"Saved full merged table to {FULL_MERGED_PARQUET}")

    del dfs, full_df
    gc.collect()

def get_parquet_columns(path: Path) -> List[str]:
    """
    Return column names from a Parquet file without reading any table data.
    Works for both pyarrow and fastparquet.
    """
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return pf.schema.names
    except ImportError:
        # Fallback: fastparquet
        from fastparquet import ParquetFile
        pf = ParquetFile(path)
        return pf.columns
    
def build_subset_table(cycle_paths: List[Path]) -> None:
    """
    Build the subset table for modeling:

    - Reads only selected columns from each per-cycle Parquet.
    - Concatenates them.
    - Enforces dtypes for continuous and categorical columns.
    - Saves subset to disk.
    """
    print("\n=== Building subset table for modeling ===")

    desired_cols = list(dict.fromkeys(EXTRA_COLUMNS + CONTINUOUS_COLS + CATEGORICAL_COLS))
    subset_dfs = []

    for p in cycle_paths:
        # Determine which of desired columns exist in this Parquet file
        print(f"  Inspecting columns in {p}")
        all_cols = get_parquet_columns(p)
        cols_this = [c for c in desired_cols if c in all_cols]

        print(f"    Reading subset columns ({len(cols_this)}): {cols_this}")
        df_sub = pd.read_parquet(p, columns=cols_this)
        subset_dfs.append(df_sub)

    subset_df = pd.concat(subset_dfs, axis=0, ignore_index=True)
    print(
        f"Combined subset shape before dtype enforcement: "
        f"{subset_df.shape[0]} rows, {subset_df.shape[1]} columns"
    )

    subset_df = enforce_dtypes(subset_df, CONTINUOUS_COLS, CATEGORICAL_COLS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save Parquet
    print(f"Saving subset Parquet to {SUBSET_PARQUET}")
    subset_df.to_parquet(SUBSET_PARQUET, index=False, engine="pyarrow")

    # And CSV
    print(f"Saving subset CSV to {SUBSET_CSV}")
    subset_df.to_csv(SUBSET_CSV, index=False)

    print("Subset table saved.")
    del subset_dfs, subset_df
    gc.collect()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Incrementally merge each cycle into its own Parquet
    cycle_parquet_paths: List[Path] = []
    for cycle in CYCLES:
        p = merge_cycle_incremental(cycle)
        cycle_parquet_paths.append(p)

    # 2. Optionally build a single big full table (can be heavy)
    if BUILD_FULL_MERGED_TABLE:
        build_full_merged_table(cycle_parquet_paths)
    else:
        print("\nSkipping building a single full merged table "
              "(BUILD_FULL_MERGED_TABLE = False)")

    # 3. Build the subset table (efficient: only reads needed columns)
    build_subset_table(cycle_parquet_paths)

    print("\nAll done.")


if __name__ == "__main__":
    main()