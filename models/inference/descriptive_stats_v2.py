"""
Timestamps: Datum, In kamer, Uit kamer, Booked Time of day

Durations: Duur kamer, Minuten afwijkend, Minuten onderplanning, Minuten overplanning

Categorical: Chirurg, Specialisme, Primair anest.type, Patiëntklasse, Kamer, Vertragingsreden, Dag vd week, In rapportering org.?

Numeric: Patiëntleeftijd, AVG_DUR_5_PREV_SUR, Number of Surgeries per Surgent, Total Surgeries per Day, perc_dev

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.matlib import empty
from pydantic_core.core_schema import none_schema
from scipy import stats
import statsmodels.api as sm
from sqlalchemy import nulls_last
from statsmodels.formula.api import ols
from statsmodels.stats.descriptivestats import describe
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

group_col = 'Dag vd week'
# target_col = 'absolute_perc_dev'
target_col = 'perc_dev'
csv_path = "/Users/bennet/Desktop/thesis/data/clean_MUMC_v3.csv"



# --------------------------------------------------
#                 LOAD & PREPARE DATA
# --------------------------------------------------

df = pd.read_csv(csv_path)
df = df[~df['Dag vd week'].isin(['za', 'zo'])]
# Filter out shift times (shift is from 07:00 to 17:30) start time
# Convert column to datetime if not already
df["In kamer"] = pd.to_datetime(df["In kamer"], errors="coerce")
shift_start = pd.to_datetime("07:00:00").time()
shift_end = pd.to_datetime("17:30:00").time()

print("Before filtering for shift:", df.shape[0], "rows")
df = df[df["In kamer"].dt.time.between(shift_start, shift_end)]
print("After filtering for shift:", df.shape[0], "rows")


print("Original data sample:")
print(df.head())

# df['absolute_perc_dev'] = (df['Minuten afwijkend'].abs() / df['Booked']) * 100
df['perc_dev'] = ((df['Minuten onderplanning'] - df['Minuten overplanning']) / df['Booked']) * 100
df['deviation'] = df['Duur kamer'] - df['Booked']

print("Engineered data sample:")
print(df.head())
print(describe(df))
original_df = df.copy()



# --------------------------------------------------
#                 FILTER OUTLIERS
# --------------------------------------------------

threshold = 3
z_scores = np.abs(stats.zscore(df[target_col]))

print(f"\nZ-scores for {target_col}:")
print(z_scores)

non_outlier_mask = z_scores < threshold
df_old = df.copy()
df_filtered = df[non_outlier_mask]

print(f"\n✅ Removed {len(df_old) - len(df_filtered)} outliers based on z-score threshold of {threshold}.")


#################################################################################
# HIGH-LEVEL SUMMARIES
#################################################################################

def high_level_summary(df):
    """
    1) Mean, Median, SD of "Duur kamer", "Minuten afwijkend", "Minuten onderplanning", "Minuten overplanning"

    2) Top 5 Chirurgs with highest average delay

    3) Top 5 Specialisms with highest average delay

    4) Top 5 Verrichtingen with highest percentage deviation

    5) Distribution of Delay per Dag vd week, Specialisme, Kamer

    6) Delay vs Patiëntleeftijd, AVG_DUR_5_PREV_SUR, Number of Surgeries per Surgent, Total Surgeries per Day
    """

    # 1) Mean, Median, SD of "Duur kamer", "Minuten afwijkend", "Minuten onderplanning", "Minuten overplanning"
    summary_stats = df[['Duur kamer', 'Minuten afwijkend', 'Minuten onderplanning', 'Minuten overplanning', 'deviation', 'perc_dev']].describe()
    print("\nSummary Statistics:")
    print(summary_stats)
    # count how often underplannedn and overplanned
    underplanned_count = df[df['Minuten onderplanning'] > 0].shape[0]
    overplanned_count = df[df['Minuten overplanning'] > 0].shape[0]
    print(f"\nCount of underplanned surgeries: {underplanned_count}")
    print(f"Count of overplanned surgeries: {overplanned_count}")

    # 1.1) Plot distribution of target: deviation
    plt.figure(figsize=(10, 5))
    sns.histplot(df['deviation'], bins=60, kde=True)
    plt.title("Distribution of Deviation")
    plt.xlabel("Deviation (minutes)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # 2) Top 5 Chirurgs with highest average delay
    top_chirurgs = df.groupby('Chirurg')['perc_dev'].mean().nlargest(5)
    print("\nTop 5 Chirurgs with highest average delay:")
    print(top_chirurgs)

    # 3) Top 5 Specialisms with highest average delay
    top_specialisms = df.groupby('Specialisme')['perc_dev'].mean().nlargest(5)
    print("\nTop 5 Specialisms with highest average delay:")
    print(top_specialisms)

    # 4) Top 5 Verrichtingen with highest percentage deviation
    top_verrichtingen = df.groupby('Verrichtingen')['perc_dev'].mean().nlargest(5)
    print("\nTop 5 Verrichtingen with highest percentage deviation:")
    print(top_verrichtingen)

    # 5) Distribution of Delay per Dag vd week, Specialisme, Kamer
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    sns.boxplot(data=df, x='Dag vd week', y='perc_dev', ax=axes[0])
    axes[0].set_title('Distribution of Delay per Dag vd week')

    sns.boxplot(data=df, x='Specialisme', y='perc_dev', ax=axes[1])
    axes[1].set_title('Distribution of Delay per Specialisme')

    sns.boxplot(data=df, x='Kamer', y='perc_dev', ax=axes[2])
    axes[2].set_title('Distribution of Delay per Kamer')

    plt.tight_layout()
    plt.show()

    # 6) Delay vs Patiëntleeftijd, AVG_DUR_5_PREV_SUR, Number of Surgeries per Surgent, Total Surgeries per Day
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.scatterplot(data=df, x='Patiëntleeftijd', y='perc_dev', ax=axes[0, 0])
    axes[0, 0].set_title('Delay vs Patiëntleeftijd')
    sns.scatterplot(data=df, x='AVG_DUR_5_PREV_SUR', y='perc_dev', ax=axes[0, 1])
    axes[0, 1].set_title('Delay vs AVG_DUR_5_PREV_SUR')
    sns.scatterplot(data=df, x='Number of Surgeries per Surgent', y='perc_dev', ax=axes[1, 0])
    axes[1, 0].set_title('Delay vs Number of Surgeries per Surgent')
    sns.scatterplot(data=df, x='Total Surgeries per Day', y='perc_dev', ax=axes[1, 1])
    axes[1, 1].set_title('Delay vs Total Surgeries per Day')
    plt.tight_layout()
    plt.show()



high_level_summary(df_filtered)


# Assume df is preloaded and cleaned
df = df.rename(columns={"Time of day": "time_of_day", "Patiëntleeftijd": "age", "Number of Surgeries per Surgent": "num_surgeries_per_surgent", "Total Surgeries per Day": "total_surgeries_per_day"})
"""
X = df[["num_surgeries_per_surgent"]]
y = df["perc_dev"]

# Unique categories
time_categories = df["time_of_day"].unique()

# Plotting setup
plt.figure(figsize=(8, 6))

for cat in time_categories:
    subset = df[df["time_of_day"] == cat]
    X_sub = subset[["num_surgeries_per_surgent"]]
    y_sub = subset["perc_dev"]

    model = GradientBoostingRegressor()
    model.fit(X_sub, y_sub)

    # Compute PDP for age
    pd_result = partial_dependence(model, X_sub, features=[0], grid_resolution=50)
    ages = pd_result["grid_values"][0]
    pdp = pd_result["average"][0]

    plt.plot(ages, pdp, label=cat)

plt.xlabel("Age")
plt.ylabel("Partial dependence: perc_dev")
plt.title("PDP of Age by Time of Day")
plt.legend()
plt.tight_layout()
plt.show()
"""

# Set a nicer theme
sns.set(style="whitegrid")
df['sqrt_perc_dev'] = np.sign(df['perc_dev']) * np.sqrt(np.abs(df['perc_dev']))


plt.figure(figsize=(10, 6))
sns.regplot(
    data=df,
    x='num_surgeries_per_surgent',
    y='sqrt_perc_dev',
    scatter_kws={'s': 50, 'alpha': 0.7},   # control point size and transparency
    line_kws={'color': 'red'}              # regression line color
)

plt.title('Number of Surgeries per Surgeon vs. Percentage Deviation', fontsize=14)
plt.xlabel('Number of Surgeries per Surgeon', fontsize=12)
plt.ylabel('Percentage Deviation', fontsize=12)
plt.tight_layout()
plt.show()
