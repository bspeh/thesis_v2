"""
Script Name: report_part1_regression.py
Author: Bennet Speh
Created: 2025-05-11

Description:
    Final report version for the MUMC dataset.
    This script implements a regression approach to analyze deviations.
"""

# --------------------------------------------------
#                  IMPORTS
# --------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.descriptivestats import describe
from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    linear_rainbow,
    acorr_ljungbox
)
from statsmodels.graphics.gofplots import qqplot
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
#               CONFIGURATION
# --------------------------------------------------

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

target_col = 'deviation'
csv_path = "/Users/bennet/Desktop/thesis/data/clean_MUMC_v4.csv"

# --------------------------------------------------
#         GLOBAL COLUMN RENAME FUNCTION
# --------------------------------------------------

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace whitespace in column names with underscores."""
    df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
    return df

# --------------------------------------------------
#           FEATURE ENGINEERING FUNCTION
# --------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features such as deviation, over/underestimation, and time of day."""

    df['perc_dev'] = ((df['Minuten_onderplanning'] - df['Minuten_overplanning']) / df['Booked']) * 100
    df['deviation'] = df['Duur_kamer'] - df['Booked']
    df['has_delay'] = df['deviation'] > 10
    df['overestimated'] = df['deviation'] < -10
    df['underestimated'] = df['deviation'] > 10

    def time_of_day_extended(row):
        if pd.isna(row["In_kamer"]):
            return None
        try:
            t = pd.to_datetime(row["In_kamer"]).time()
        except Exception:
            return None
        if pd.to_datetime("06:00:00").time() <= t < pd.to_datetime("10:00:00").time():
            return "06-10"
        elif pd.to_datetime("10:00:00").time() <= t < pd.to_datetime("14:00:00").time():
            return "10-14"
        elif pd.to_datetime("14:00:00").time() <= t < pd.to_datetime("18:00:00").time():
            return "14-18"
        return None

    df['Time_of_day_EXTENDED'] = df.apply(time_of_day_extended, axis=1)

    return df

# --------------------------------------------------
#              LOAD & PREPARE DATA
# --------------------------------------------------

df = pd.read_csv(csv_path)
full_df = df.copy()

# Remove weekends
df = df[~df['Dag vd week'].isin(['za', 'zo'])]

# Convert to datetime and filter shift times (07:00 to 17:30)
df["In kamer"] = pd.to_datetime(df["In kamer"], errors="coerce")
shift_start = pd.to_datetime("07:00:00").time()
shift_end = pd.to_datetime("17:30:00").time()
df = df[df["In kamer"].dt.time.between(shift_start, shift_end)]

# Rename columns to use underscores
df = rename_columns(df)

# Apply feature engineering
df = engineer_features(df)

# Show sample
print("Engineered data sample:")
print(df.head())
print(describe(df))

original_df = df.copy()

# --------------------------------------------------
#              FILTER OUTLIERS
# --------------------------------------------------

print("\n\nFiltering outliers...")
threshold = 3
z_scores = np.abs(stats.zscore(df[target_col]))
df_filtered = df[z_scores < threshold]

print(f"\n✅ Removed {len(df) - len(df_filtered)} outliers based on z-score threshold of {threshold}.")
df = df_filtered.copy()
print(df.head())
print("\n\nLength of original dataframe:", len(original_df))
print("Length of filtered dataframe:", len(df))


# --------------------------------------------------
#                 OLS GLOBAL
# --------------------------------------------------
