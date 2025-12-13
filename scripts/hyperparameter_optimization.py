import pandas as pd
import numpy as np
from typing import List
import argparse
import warnings
import numpy as np
import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.utils import load_synthesizer
from sdv.sampling import Condition

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

dat = pd.read_csv('data/ukb/OSA_test2.csv')
print(dat['OSA'].value_counts())

dat = dat.drop(columns=['f_20023_0_0', 'f_20023_1_0', 'f_4282_0_0', 'f_4282_1_0', 'Date', 'f_20016_0_0', 'f_20016_1_0', "f_eid", "IID"])
dat = dat.drop(columns=[col for col in dat.columns if '_3_0' in col])
# dat = dat.dropna()
print(dat.info())
print(dat['OSA'].value_counts())

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def clean_missing_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert UKB-style codes for 'Refused', 'Don't know', or 'Missing'
    to NaN for numeric columns, and certain strings to NaN for object columns.

    This does NOT try to interpret 88/888 etc., only classic 7/9 patterns.
    """
    missing_codes = {
        7, 9,
        77, 99,
        777, 999,
        7777, 9999,
        77777, 99999,
    }

    # Numeric columns: map code values to NaN
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        df[col] = df[col].mask(df[col].isin(missing_codes))

    # String columns: common literal labels
    text_missing = {"Refused", "REFUSED", "Don't know", "DON'T KNOW", "Missing", "MISSING"}
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].replace(list(text_missing), np.nan)

    return df


def train_sdv_synthesizer(df: pd.DataFrame, 
                          epoch: int=300, 
                          batch_size: int=500, 
                          lr: float=1e-4,
                          random_state: int = 42) -> CTGANSynthesizer:
    """
    Train an SDV CTGAN synthesizer on the dataframe, treating 'osa_diag' as
    a conditional variable for later balanced sampling.
    """
    if "OSA" not in df.columns:
        raise ValueError("'OSA' column must exist in dataframe before training.")
    # Optionally drop unique identifier columns (e.g., SEQN) to avoid overfitting IDs
    # df = df.drop(columns=["f_eid", "IID"])
    
    # Build metadata automatically
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # Ensure osa and date columns are treated as categorical / datetime
    metadata.update_column(
        column_name="OSA",
        sdtype="categorical",
    )
    for col_id in ["centre"]:
        metadata.update_column(
            column_name=col_id,
            sdtype="datetime",
            datetime_format="%Y-%m-%d"
        )

    # Create and train synthesizer
    synthesizer = CTGANSynthesizer(
        metadata,
        enforce_rounding=True,
        epochs=epoch,
        batch_size=batch_size,
        discriminator_lr = lr,
        generator_lr = lr,
        verbose=True,
        cuda=True,        # set to True if you want to use GPU and have one
    )
    # handle missing values automatically
    # has to make a function for this
    
    synthesizer.fit(df)

    return synthesizer


def generate_balanced_samples(
    synthesizer,
    n_samples: int = 40000,
    label_col: str = "OSA",
):
    """
    Generate synthetic samples with equal numbers of 0 and 1 in `label_col`
    using SDV conditional sampling.
    """
    if n_samples % 2 != 0:
        raise ValueError("n_samples must be even to generate a balanced dataset.")

    half = n_samples // 2

    # One condition for label = 0, one for label = 1
    cond_0 = Condition(
        num_rows=half,
        column_values={label_col: 0},
    )
    cond_1 = Condition(
        num_rows=half,
        column_values={label_col: 1},
    )

    synthetic = synthesizer.sample_from_conditions(
        conditions=[cond_0, cond_1]
        # you can also tweak:
        # batch_size=...,
        # max_tries_per_batch=...,
    )

    return synthetic

random_state = 42
lr = 5e-5
batch_size = 1000
synthesizer_file = f'results/ukb_synthesizer_osa_{lr}_{batch_size}.pkl'
# -------------------------------------------------------------------------
# 4. Train SDV synthesizer (CTGAN) with osa_diag as conditional variable
# -------------------------------------------------------------------------
print("Training SDV CTGAN synthesizer (conditional on OSA)...")
np.random.seed(random_state)

synthesizer = train_sdv_synthesizer(dat, epoch = 600, lr=lr, batch_size=batch_size, random_state=random_state)

fig = synthesizer.get_loss_values_plot()
fig.show()
with open(f'results/ukb_synthesizer_osa_loss_{lr}_{batch_size}.png', 'wb') as f:
    f.write(fig.to_image(format='png', width=1600, height=800))

synthesizer.save(synthesizer_file)
print(f"Synthesizer saved to {synthesizer_file}")
