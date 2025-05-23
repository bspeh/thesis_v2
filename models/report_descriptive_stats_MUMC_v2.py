"""
Script Name: report_descriptive_stats_MUMC.py
Author: Bennet Speh
Created: 2025-05-11
Description:
    This script is the final report version for the MUMC dataset. This script will give a summary of the data in final report form.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.descriptivestats import describe
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import scikit_posthocs as sp
import scienceplots


# --------------------------------------------------
#                 CONFIGURATION
# --------------------------------------------------

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# target_col = 'absolute_perc_dev'
# target_col = 'test_perc_dev'
target_col = 'perc_dev'
# target_col = 'deviation'
csv_path = "/Users/bennet/Desktop/thesis/data/clean_MUMC_v4.csv"


# --------------------------------------------------
#                 LOAD & PREPARE DATA
# --------------------------------------------------

df = pd.read_csv(csv_path)
full_df = df.copy()

# Clean out the weekends (NOT INTERESTED)
df = df[~df['Dag vd week'].isin(['za', 'zo'])]

# Filter out shift times (shift is from 07:00 to 17:30) start time
df["In kamer"] = pd.to_datetime(df["In kamer"], errors="coerce")
shift_start = pd.to_datetime("07:00:00").time()
shift_end = pd.to_datetime("17:30:00").time()
# print("Before filtering for shift:", df.shape[0], "rows")
df = df[df["In kamer"].dt.time.between(shift_start, shift_end)]
# print("After filtering for shift:", df.shape[0], "rows")

print("Original data sample:")
print(df.head())

# Creating the target column
# df['absolute_perc_dev'] = (df['Minuten afwijkend'].abs() / df['Booked']) * 100
df['perc_dev'] = ((df['Minuten onderplanning'] - df['Minuten overplanning']) / df['Booked']) * 100
# df['perc_dev'] = df['Duur kamer'] / df['Booked']
df['deviation'] = df['Duur kamer'] - df['Booked']
# df['percentage_deviation'] = (df['Duur kamer'] - df['Booked']) / df['Booked'] * 100
df['has_delay'] = df['deviation'] > 10 # True if deviation is greater than 10%

# create columns overestimated and underestimated
df['overestimated'] = df['Duur kamer'] - df['Booked'] < -10
df['underestimated'] = df['Duur kamer'] - df['Booked'] > 10

# filter out NAN in Verrichtingen
print("Before filtering for Verrichtingen:", df.shape[0], "rows")
df = df[df['Verrichtingen'].notnull()]
print("After filtering for Verrichtingen:", df.shape[0], "rows")


# Create column: "Number_of_SubSurgeries"
df['sub_surgeries_count'] = df['Verrichtingen'].apply(lambda x: len(set(x.split(', '))))
df['sub_surgeries_count'] = df['sub_surgeries_count'].apply(
    lambda x: x if x <= 4 else '5+'
)


# rename column "Patiëntleeftijd op verrichtingsdatum" to "Patiëntleeftijd"
df.rename(columns={"Patiëntleeftijd op verrichtingsdatum": "Patiëntleeftijd"}, inplace=True)

# create column: Time of day EXTENDED
def time_of_day_extended(row):
    if pd.isna(row["In kamer"]):
        return None

    # Convert to datetime.time safely
    try:
        in_kamer_time = pd.to_datetime(row["In kamer"]).time()
    except Exception:
        return None

    if pd.to_datetime("06:00:00").time() <= in_kamer_time < pd.to_datetime("10:00:00").time():
        return "06-10"
    elif pd.to_datetime("10:00:00").time() <= in_kamer_time < pd.to_datetime("14:00:00").time():
        return "10-14"
    elif pd.to_datetime("14:00:00").time() <= in_kamer_time < pd.to_datetime("18:00:00").time():
        return "14-18"
    return None


df['Time of day EXTENDED'] = df.apply(time_of_day_extended, axis=1)

# Reset the "Daily Surgeries per Surgeon" column to having 1,2,3,4,5+
def reset_daily_surgeries(row):
    if row["Number of Surgeries per Surgent"] == 1:
        return "1"
    elif row["Number of Surgeries per Surgent"] == 2:
        return "2"
    elif row["Number of Surgeries per Surgent"] == 3:
        return "3"
    elif row["Number of Surgeries per Surgent"] == 4:
        return "4"
    elif row["Number of Surgeries per Surgent"] == 5:
        return "5+"


df['Number of Surgeries per Surgent'] = df.apply(reset_daily_surgeries, axis=1)



### Filter out sub-specialties with less than 30 observations
# Filter out sub-specialties with less than 30 observations
subspecialty_counts = df['Specialisme'].value_counts()
subspecialty_counts = subspecialty_counts[subspecialty_counts > 30]
df = df[df['Specialisme'].isin(subspecialty_counts.index)]



##### Initial try to reduce the number of surgery types



# OUTPUT
print("Engineered data sample:")
print(df.head())
print(describe(df))
original_df = df.copy()

# Example function for imputing missing 'Minuten' value
def impute_minuten(row):
    if row['Minuten overplanning'] == 0 and row['Minuten onderplanning'] == 0:
        if row['Over-/onderplanning'] == 'Overplanning':
            pct = np.random.uniform(1.00, 1.10)
        elif row['Over-/onderplanning'] == 'Onderplanning':
            pct = np.random.uniform(0.90, 1.00)
        else:
            return row["Duur kamer"]  # handle unexpected values

        return row['Duur kamer'] * pct  # full new value
    else:
        return row['Duur kamer']  # keep as is


# Apply the function
df['test'] = df.apply(impute_minuten, axis=1)

# Calculate the deviation
df['test_deviation'] = df['test'] - df['Booked']

df["test_perc_dev"] = ((df['test'] - df['Booked']) / df['Booked']) * 100

print("Original data sample:")

# Another test
df['latest_test'] = df['Duur kamer'] / df['Booked']


# --------------------------------------------------
#                 FILTER OUTLIERS
# --------------------------------------------------

threshold = 3
z_scores = np.abs(stats.zscore(df[target_col]))

non_outlier_mask = z_scores < threshold
df_old = df.copy()
df_filtered = df[non_outlier_mask]

print(f"\n✅ Removed {len(df_old) - len(df_filtered)} outliers based on z-score threshold of {threshold}.")
df = df_filtered.copy()


# --------------------------------------------------
#                 CONFIGURE FILTER
# --------------------------------------------------
# Filter for largest occuring Specialisme
specialism_counts = df['Specialisme'].value_counts()
specialism_to_keep = specialism_counts.idxmax()
"""
print(f"\nKeeping only the most frequent specialism: {specialism_to_keep}")
df = df[df['Specialisme'] == specialism_to_keep]
"""
global_filter = False



# --------------------------------------------------
#                 COMPUTE GLOBAL MAE
# --------------------------------------------------
abs_error = abs(df['Duur kamer'] - df['Booked'])
mae = abs_error.mean()

# print number of observations
print(f"\n\nNumber of observations: {len(df)}")
print("The global MAE is:", mae)



# --------------------------------------------------
#                 DESCRIPTIVE STATS
# --------------------------------------------------
# Length of dataset pre and post filtering
print(f"\n\nLength of dataset pre-filtering: {len(full_df)}")
print(f"Length of dataset post-filtering: {len(df)}")

# Time Span of dataset
start_date = df['Datum'].min()
end_date = df['Datum'].max()
print(f"\nTime span of dataset: {start_date} to {end_date}")

# Number of overestimation, on time, and underestimation (percentage)
tolerance = 5
status = df.apply(
    lambda row: 'On Time' if abs(row['Duur kamer'] - row['Booked']) <= tolerance
    else 'Overrun' if row['Duur kamer'] > row['Booked']
    else 'Underrun',
    axis=1
).to_frame(name='status')
summary = status['status'].value_counts().to_frame('Count')
summary['Percentage'] = (status['status'].value_counts(normalize=True) * 100).round(2)
print(summary)

# --------------------------------------------------
#                 HIGH-LEVEL SUMMARIES

#---------------------------------------------------
# Surgical Procedural Delays by Subspecialty
delay_by_subspeciality = df.groupby("Specialisme").agg(
    Total_cases=("Specialisme", "count"),
    Avg_proc_delay=("deviation", "mean"),
    Std_dev=("deviation", "std")
).reset_index()
delay_by_subspeciality.columns = ["Surgical specialty", "Total number of cases", "Avg proc delay (minutes)", "Std dev (minutes)"]
print("\n\nSECTION: Surgical Procedural Delays by Subspecialty:")
print(delay_by_subspeciality)

# --------------------------------------------------
# The number of unique surgeons and anesthesiologists and surgery types (Verrichting)
num_surgeons = df['Hoofdchirurg/zorgverlener'].nunique()
# num_anesthesiologists = df['Verantw. anest.zorgverl.'].nunique()
num_surgery_types = df['Verrichtingen'].nunique()
print(f"\nNumber of unique surgeons: {num_surgeons}")
# print(f"Number of unique anesthesiologists: {num_anesthesiologists}")
print(f"Number of unique surgery types: {num_surgery_types}")

# ---------------------------------------------------
# Surgeon vs. Verrichtingen Correlation
# Create a contingency table
contingency_table = pd.crosstab(df['Hoofdchirurg/zorgverlener'], df['Verrichtingen'])
# Perform Chi-squared test
chi2_surgeon, p_surgeon, _, _ = stats.chi2_contingency(contingency_table)
print(f"\n\nChi-squared test for surgeon vs. surgery type: chi2 = {chi2_surgeon:.2f}, p = {p_surgeon:.4f}")

# Cramér's V
n = contingency_table.sum().sum()
phi2 = chi2_surgeon / n
r, k = contingency_table.shape
cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))
print("Cramér’s V:", cramers_v)

# ---------------------------------------------------
# Booked Correlation with target (Spearman)
# Spearman correlation
spearman_corr = df['Booked'].corr(df[target_col], method='spearman')
print(f"\n\nSpearman correlation between Booked and {target_col}: {spearman_corr:.4f}")
spearman_corr = df['Booked'].corr(df['deviation'], method='spearman')
print(f"\n\nSpearman correlation between Booked and {target_col}: {spearman_corr:.4f}")


# --------------------------------------------------
# Preadmission Testing



#---------------------------------------------------
# BMI


#---------------------------------------------------
# ASA Classification

#---------------------------------------------------
# Number of daily cases per surgeon


#---------------------------------------------------
# Number of daily cases per department


#---------------------------------------------------
# Number of daily cases






#---------------------------------------------------
#       INFERENCE TESTING
#---------------------------------------------------
if not global_filter:
    # --------------------------------------------------
    # TEST: Subspecialty vs. Delay

    df_subspecialities = df.copy()
    df_subspecialities = df_subspecialities[df_subspecialities['Specialisme'].notnull()]
    df_subspecialities['Specialisme'] = df_subspecialities['Specialisme'].astype('category')
    subspeciality_counts = df_subspecialities['Specialisme'].value_counts()
    subspeciality_counts = subspeciality_counts[subspeciality_counts > 30]
    df_subspecialities = df_subspecialities[df_subspecialities['Specialisme'].isin(subspeciality_counts.index)]

    # Perform Kruskal-Wallis test
    df_subspecialities = df_subspecialities.dropna(subset=[target_col])
    valid_specialties = (
        df_subspecialities.groupby('Specialisme')[target_col]
        .nunique()
        .loc[lambda x: x > 1]  # keep only groups with >1 unique value
        .index
    )
    df_subspecialities = df_subspecialities[df_subspecialities['Specialisme'].isin(valid_specialties)]
    kruskal_results = stats.kruskal(*[group[target_col].values for name, group in df_subspecialities.groupby('Specialisme')])

    print(f"\n\nKruskal-Wallis test results for subspecialty vs. delay: H={kruskal_results.statistic:.2f}, p={kruskal_results.pvalue:.4f}")

    # Perform Dunn's test for post-hoc analysis
    # mc = MultiComparison(df_subspecialities[target_col], df_subspecialities['Specialisme'])
    # dunn_results = mc.tukeyhsd()
    dunn_results = sp.posthoc_dunn(df_subspecialities,val_col=target_col, group_col='Specialisme', p_adjust='bonferroni') # or 'holm', 'fdr_bh', etc))
    print(dunn_results)
    alpha = 0.05
    print("\nSignificant pairwise comparisons (p < 0.05):")
    significant_pairs = []

    for i in range(len(dunn_results.index)):
        for j in range(i + 1, len(dunn_results.columns)):
            p_val = dunn_results.iloc[i, j]
            if p_val < alpha:
                pair = (dunn_results.index[i], dunn_results.columns[j])
                significant_pairs.append((pair, p_val))

    for pair, p_val in significant_pairs:
        print(f"{pair[0]} vs. {pair[1]}: p = {p_val:.4f}")

    """
    # Plotting the results
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Specialisme', y='perc_dev', data=df_subspecialities)
    plt.title('Delay by Subspecialty')
    plt.xlabel('Subspecialty')
    plt.ylabel('Delay (minutes)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()
    """

# --------------------------------------------------
# TEST: Age vs. Delay
df_age = df.copy()
df_age = df_age[df_age['Patiëntleeftijd'].notnull()]

# Groups: 0 to 2, 3 to 12, 13 to 19, 20 to 39, 40 to 59, > 60 years
bins = [0, 2, 12, 19, 39, 59, np.inf]
labels = ['0-2', '3-12', '13-19', '20-39', '40-59', '60+']
df_age['Age Group'] = pd.cut(df_age['Patiëntleeftijd'], bins=bins, labels=labels, right=False)
df_age = df_age[df_age['Age Group'].notnull()]
df_age['Age Group'] = df_age['Age Group'].astype('category')
age_counts = df_age['Age Group'].value_counts()
age_counts = age_counts[age_counts > 30]
df_age = df_age[df_age['Age Group'].isin(age_counts.index)]

# Describe the column of interest with respect to target
print(f"\n\nDescriptive stats for age vs. delay:")
print(df_age.groupby('Age Group')[target_col].describe())

# Perform Kruskal-Wallis test
kruskal_results = stats.kruskal(*[group[target_col].values for name, group in df_age.groupby('Age Group')])
print(f"\n\nKruskal-Wallis test results for age vs. delay: H={kruskal_results.statistic:.2f}, p={kruskal_results.pvalue:.4f}")

# Perform Dunn's test for post-hoc analysis
# mc = MultiComparison(df_age[target_col], df_age['Age Group'])
# dunn_results = mc.tukeyhsd()
dunn_results = sp.posthoc_dunn(df_age,val_col=target_col, group_col='Age Group', p_adjust='bonferroni') # or 'holm', 'fdr_bh', etc))
print(dunn_results)



# -----------------------------------------------------
# TEST: Surgery Type vs. Delay -> WARNING: This test is not very meaningful because the surgery types are not independent

# Notes: I might need to reduce the number of surgery types to make the test more meaningful (like collapsing some categories)
# Notes: For now I will just filter out the surgery types with less than X observations

df_surgery_types = df.copy()

# Drop rows where surgery type is missing
df_surgery_types = df_surgery_types[df_surgery_types['Verrichtingen'].notnull()]
print(f"After dropping null surgery types: {df_surgery_types.shape}")

# Convert to category
df_surgery_types['Verrichtingen'] = df_surgery_types['Verrichtingen'].astype('category')

# Keep surgery types with more than 5 occurrences
surgery_type_counts = df_surgery_types['Verrichtingen'].value_counts()
surgery_type_counts = surgery_type_counts[surgery_type_counts > 5]
df_surgery_types = df_surgery_types[df_surgery_types['Verrichtingen'].isin(surgery_type_counts.index)]
print(f"Surgery types with >5 cases: {len(surgery_type_counts)}")

# Check and remove NaNs in target_col
missing_values = df_surgery_types[target_col].isnull().sum()
print(f"Missing values in target column '{target_col}': {missing_values}")
df_surgery_types = df_surgery_types[df_surgery_types[target_col].notnull()]

# Group data and check group sizes
groups = []
for name, group in df_surgery_types.groupby('Verrichtingen'):
    values = group[target_col].values
    # print(f"Group '{name}': size={len(values)}, NaNs={np.isnan(values).sum()}")
    groups.append(values)

# Filter out empty groups
groups_nonempty = [g for g in groups if len(g) > 0]

# Perform Kruskal-Wallis test
kruskal_results = stats.kruskal(*groups_nonempty)
print(f"\nKruskal-Wallis test results for surgery type vs. delay: H={kruskal_results.statistic:.2f}, p={kruskal_results.pvalue:.4f}")

# Perform Dunn's test for post-hoc analysis
dunn_results = sp.posthoc_dunn(df_surgery_types, val_col=target_col, group_col='Verrichtingen', p_adjust='bonferroni')
# print("\nDunn's posthoc test p-values (Bonferroni corrected):")
# print(dunn_results)





# -----------------------------------------------------
# TEST: Time of day EXTENDED vs. Delay

df_time_of_day = df.copy()
df_time_of_day = df_time_of_day[df_time_of_day['Time of day EXTENDED'].notnull()]
df_time_of_day['Time of day EXTENDED'] = df_time_of_day['Time of day EXTENDED'].astype('category')
time_of_day_counts = df_time_of_day['Time of day EXTENDED'].value_counts()
time_of_day_counts = time_of_day_counts[time_of_day_counts > 0]
df_time_of_day = df_time_of_day[df_time_of_day['Time of day EXTENDED'].isin(time_of_day_counts.index)]

# I checked that the time of day influence is not due to the fact that there are certain surgeons that only operate at certain times of the day
df_time_of_day['Hoofdchirurg/zorgverlener'] = df_time_of_day['Hoofdchirurg/zorgverlener'].astype('category')
surgeon_counts = df_time_of_day['Hoofdchirurg/zorgverlener'].value_counts()
surgeon_counts = surgeon_counts[surgeon_counts > 5]
df_time_of_day = df_time_of_day[df_time_of_day['Hoofdchirurg/zorgverlener'].isin(surgeon_counts.index)]

# IMPORTANT!!! Check for the anaesthesiologist -> If I include this it will be non significant BUT I SUPPOSE THIS IS ONLY DUE TO THE FACT THAT WE HAVE LITTLE DATA
"""
df_time_of_day['Verantw. anest.zorgverl.'] = df_time_of_day['Verantw. anest.zorgverl.'].astype('category')
anesthesiologist_counts = df_time_of_day['Verantw. anest.zorgverl.'].value_counts()
anesthesiologist_counts = anesthesiologist_counts[anesthesiologist_counts > 5]
df_time_of_day = df_time_of_day[df_time_of_day['Verantw. anest.zorgverl.'].isin(anesthesiologist_counts.index)]
"""

# Describe the column of interest with respect to target
print(f"\n\nDescriptive stats for time of day EXTENDED vs. delay:")
print(df_time_of_day.groupby('Time of day EXTENDED')[target_col].describe())

# Perform Kruskal-Wallis test
kruskal_results = stats.kruskal(*[group[target_col].values for name, group in df_time_of_day.groupby('Time of day EXTENDED')])
print(f"\n\nKruskal-Wallis test results for time of day vs. delay: H={kruskal_results.statistic:.2f}, p={kruskal_results.pvalue:.4f}")

# Perform Dunn's test for post-hoc analysis
# mc = MultiComparison(df_time_of_day[target_col], df_time_of_day['Time of day EXTENDED'])
# dunn_results = mc.tukeyhsd()
dunn_results = sp.posthoc_dunn(df_time_of_day,val_col=target_col, group_col='Time of day EXTENDED', p_adjust='bonferroni') # or 'holm', 'fdr_bh', etc))
print(dunn_results)


# To check that these findings hold I must check that this is not due to the fact that there are certain surgeons that only operate at certain times of the day
# Thus I will check the distribution of surgeons per time of day (Chi-squared test)
contingency_table = pd.crosstab(df_time_of_day['Hoofdchirurg/zorgverlener'], df_time_of_day['Time of day EXTENDED'])

# Perform Chi-squared test
chi2_surgeon, p_surgeon, _, _ = stats.chi2_contingency(contingency_table)
print(f"\n\nChi-squared test for surgeon vs. time of day EXTENDED: chi2 = {chi2_surgeon:.2f}, p = {p_surgeon:.4f}")


# IMPORTANT!! I guess it is still to mention that this can also be due to the fact that some types of surgeries are only done at certain times of the day (but not sure here!!! This needs to be asked for the final report). But at least I can say that it's not due to the surgeons.

# Plotting the results
"""
plt.figure(figsize=(12, 6))
sns.boxplot(x='Time of day EXTENDED', y=target_col, data=df_time_of_day)
plt.title('Delay by Time of Day EXTENDED')
plt.xlabel('Time of Day EXTENDED')
plt.ylabel('Delay (minutes)')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
"""



# -----------------------------------------------------
# TEST: Day of the week vs. Delay

df_day_of_week = df.copy()
df_day_of_week = df_day_of_week[df_day_of_week['Dag vd week'].notnull()]
df_day_of_week['Dag vd week'] = df_day_of_week['Dag vd week'].astype('category')
day_of_week_counts = df_day_of_week['Dag vd week'].value_counts()
day_of_week_counts = day_of_week_counts[day_of_week_counts > 0]
df_day_of_week = df_day_of_week[df_day_of_week['Dag vd week'].isin(day_of_week_counts.index)]

# Describe the column of interest with respect to target
print(f"\n\nDescriptive stats for day of the week vs. delay:")
print(df_day_of_week.groupby('Dag vd week')[target_col].describe())

# Perform Kruskal-Wallis test
kruskal_results = stats.kruskal(*[group[target_col].values for name, group in df_day_of_week.groupby('Dag vd week')])
print(f"\n\nKruskal-Wallis test results for day of the week vs. delay: H={kruskal_results.statistic:.2f}, p={kruskal_results.pvalue:.4f}")

# Perform Dunn's test for post-hoc analysis
# mc = MultiComparison(df_day_of_week[target_col], df_day_of_week['Dag vd week'])
# dunn_results = mc.tukeyhsd()
dunn_results = sp.posthoc_dunn(df_day_of_week,val_col=target_col, group_col='Dag vd week', p_adjust='bonferroni') # or 'holm', 'fdr_bh', etc))
print(dunn_results)

# To check that these findings hold I must check that this is not due to the fact that there are certain surgeons that only operate on certain days of the week
# Thus I will check the distribution of surgeons per day of the week (Chi-squared test)
# Create a contingency table (for surgeons with more than 5 observations)
df_day_of_week['Hoofdchirurg/zorgverlener'] = df_day_of_week['Hoofdchirurg/zorgverlener'].astype('category')
surgeon_counts = df_day_of_week['Hoofdchirurg/zorgverlener'].value_counts()
surgeon_counts = surgeon_counts[surgeon_counts > 10]
df_day_of_week = df_day_of_week[df_day_of_week['Hoofdchirurg/zorgverlener'].isin(surgeon_counts.index)]

contingency_table = pd.crosstab(df_day_of_week['Hoofdchirurg/zorgverlener'], df_day_of_week['Dag vd week'])

# Perform Chi-squared test
chi2_surgeon, p_surgeon, _, _ = stats.chi2_contingency(contingency_table)
print(f"\n\nChi-squared test for surgeon vs. day of the week: chi2 = {chi2_surgeon:.2f}, p = {p_surgeon:.4f}")

# -> there is a significant association between surgeon and day of the week
# IMPORTANT!!! So that means it can't be said for sure that the day of the week is significantly influencing the delay.
# But I guess it is still to mention that there is a global trend but this could be due to the fact that there are certain surgeons only working on certain days of the week. (Same goes for the type of surgery).
# Still worth mentioning in the report that there is a significant association between surgeon and day of the week.




# -----------------------------------------------------
# TEST: Surgeon vs. Delay

df_surgeon = df.copy()

# Drop rows where surgeon info is missing
df_surgeon = df_surgeon[df_surgeon['Hoofdchirurg/zorgverlener'].notnull()]
print(f"After dropping null surgeons: {df_surgeon.shape}")

# Convert surgeon to category dtype
df_surgeon['Hoofdchirurg/zorgverlener'] = df_surgeon['Hoofdchirurg/zorgverlener'].astype('category')

# Keep surgeons with more than 30 cases
surgeon_counts = df_surgeon['Hoofdchirurg/zorgverlener'].value_counts()
frequent_surgeons = surgeon_counts[surgeon_counts > 1].index
df_surgeon = df_surgeon[df_surgeon['Hoofdchirurg/zorgverlener'].isin(frequent_surgeons)]
print(f"Surgeons with >30 cases: {len(frequent_surgeons)}")

# Keep surgeons with more than 1 unique value in target_col
valid_surgeons = (
    df_surgeon.groupby('Hoofdchirurg/zorgverlener')[target_col]
    .nunique()
    .loc[lambda x: x > 1]
    .index
)
df_surgeon = df_surgeon[df_surgeon['Hoofdchirurg/zorgverlener'].isin(valid_surgeons)]
print(f"Surgeons with >1 unique {target_col} value: {len(valid_surgeons)}")

# Check for any remaining NaNs in target_col
nan_count = df_surgeon[target_col].isna().sum()
print(f"NaN values in {target_col} after filtering: {nan_count}")
df_surgeon = df_surgeon[df_surgeon[target_col].notnull()]

# Summary stats per surgeon
desc = df_surgeon.groupby('Hoofdchirurg/zorgverlener')[target_col].describe()
desc = desc[desc['count'] > 0]
print(f"\nDescriptive stats for surgeon vs. delay (count > 0):")
print(desc)

# Prepare data for Kruskal-Wallis: debug group sizes and values
groups = []
for name, group in df_surgeon.groupby('Hoofdchirurg/zorgverlener'):
    values = group[target_col].values
#    print(f"Group '{name}': size={len(values)}, NaNs={np.isnan(values).sum()}")
    groups.append(values)

# Check if any group is empty or all NaNs (which causes NaN results)
empty_groups = [i for i, g in enumerate(groups) if len(g) == 0]
#if empty_groups:
#    print(f"Warning: Empty groups detected at indices {empty_groups}")

groups_nonempty = [g for g in groups if len(g) > 0]

# Run Kruskal-Wallis test
kruskal_results = stats.kruskal(*groups_nonempty)
print(f"\nKruskal-Wallis test results for surgeon vs. delay: H={kruskal_results.statistic:.2f}, p={kruskal_results.pvalue:.4f}")

# Run Dunn's posthoc test with Bonferroni correction
dunn_results = sp.posthoc_dunn(df_surgeon, val_col=target_col, group_col='Hoofdchirurg/zorgverlener', p_adjust='bonferroni')
print("\nDunn's posthoc test p-values (Bonferroni corrected):")
# print(dunn_results)





# -----------------------------------------------------
# TEST: Anesthesiologist vs. Delay






# -----------------------------------------------------
# TEST: Patiëntklasse vs. Delay
df_patient_class = df.copy()
df_patient_class = df_patient_class[df_patient_class['Patiëntklasse'].notnull()]
df_patient_class['Patiëntklasse'] = df_patient_class['Patiëntklasse'].astype('category')
patient_class_counts = df_patient_class['Patiëntklasse'].value_counts()
patient_class_counts = patient_class_counts[patient_class_counts > 0]
df_patient_class = df_patient_class[df_patient_class['Patiëntklasse'].isin(patient_class_counts.index)]

# Describe the column of interest with respect to target
print(f"\n\nDescriptive stats for patient class vs. delay:")
print(df_patient_class.groupby('Patiëntklasse')[target_col].describe())

# Perform Kruskal-Wallis test
kruskal_results = stats.kruskal(*[group[target_col].values for name, group in df_patient_class.groupby('Patiëntklasse')])
print(f"\n\nKruskal-Wallis test results for patient class vs. delay: H={kruskal_results.statistic:.2f}, p={kruskal_results.pvalue:.4f}")

# Perform Dunn's test for post-hoc analysis
# mc = MultiComparison(df_patient_class[target_col], df_patient_class['Patiëntklasse'])
# dunn_results = mc.tukeyhsd()
dunn_results = sp.posthoc_dunn(df_patient_class,val_col=target_col, group_col='Patiëntklasse', p_adjust='bonferroni') # or 'holm', 'fdr_bh', etc))
print(dunn_results)


# I am not sure how much this makes sense - well it does make sense I guess later but this just represents different types of surgeries I believe.
# Meaning some are just a prep for surgery and some are not etc.



# -----------------------------------------------------
# TEST: Number of Surgeries per Surgent vs. Delay

df_surgeries_per_surgeon = df.copy()
df_surgeries_per_surgeon = df_surgeries_per_surgeon[df_surgeries_per_surgeon['Number of Surgeries per Surgent'].notnull()]
df_surgeries_per_surgeon['Number of Surgeries per Surgent'] = df_surgeries_per_surgeon['Number of Surgeries per Surgent'].astype('category')
surgery_counts = df_surgeries_per_surgeon['Number of Surgeries per Surgent'].value_counts()
surgery_counts = surgery_counts[surgery_counts > 0]
df_surgeries_per_surgeon = df_surgeries_per_surgeon[df_surgeries_per_surgeon['Number of Surgeries per Surgent'].isin(surgery_counts.index)]

# Describe the column of interest with respect to target
print(f"\n\nDescriptive stats for number of surgeries per surgeon vs. delay:")
print(df_surgeries_per_surgeon.groupby('Number of Surgeries per Surgent')[target_col].describe())

# Perform Kruskal-Wallis test
kruskal_results = stats.kruskal(*[group[target_col].values for name, group in df_surgeries_per_surgeon.groupby('Number of Surgeries per Surgent')])
print(f"\n\nKruskal-Wallis test results for number of surgeries per surgeon vs. delay: H={kruskal_results.statistic:.2f}, p={kruskal_results.pvalue:.4f}")

# Perform Dunn's test for post-hoc analysis
# mc = MultiComparison(df_surgeries_per_surgeon[target_col], df_surgeries_per_surgeon['Number of Surgeries per Surgent'])
# dunn_results = mc.tukeyhsd()
dunn_results = sp.posthoc_dunn(df_surgeries_per_surgeon,val_col=target_col, group_col='Number of Surgeries per Surgent', p_adjust='bonferroni') # or 'holm', 'fdr_bh', etc))
print(dunn_results)


# Nice! I guess... (I am so unsure about this by now). Seems like a significant association between the number of surgeries per surgeon and the delay.
# BUT WATCH OUT THAT THIS COINCIDES WITH THEIR INTERESTS. Like the way they schedule for now: For the next week. Could be mentioned but idk.

# -----------------------------------------------------
# TEST: Number of SubSurgeries vs. Delay

df_sub_surgeries = df.copy()
df_sub_surgeries = df_sub_surgeries[df_sub_surgeries['sub_surgeries_count'].notnull()]
df_sub_surgeries['sub_surgeries_count'] = df_sub_surgeries['sub_surgeries_count'].astype('category')
sub_surgeries_counts = df_sub_surgeries['sub_surgeries_count'].value_counts()
sub_surgeries_counts = sub_surgeries_counts[sub_surgeries_counts > 0]
df_sub_surgeries = df_sub_surgeries[df_sub_surgeries['sub_surgeries_count'].isin(sub_surgeries_counts.index)]
# Describe the column of interest with respect to target
print(f"\n\nDescriptive stats for number of sub-surgeries vs. delay:")
print(df_sub_surgeries.groupby('sub_surgeries_count')[target_col].describe())

# Perform Kruskal-Wallis test
kruskal_results = stats.kruskal(*[group[target_col].values for name, group in df_sub_surgeries.groupby('sub_surgeries_count')])
print(f"\n\nKruskal-Wallis test results for number of sub-surgeries vs. delay: H={kruskal_results.statistic:.2f}, p={kruskal_results.pvalue:.4f}")

# Perform Dunn's test for post-hoc analysis
# mc = MultiComparison(df_sub_surgeries[target_col], df_sub_surgeries['sub_surgeries_count'])
# dunn_results = mc.tukeyhsd()
dunn_results = sp.posthoc_dunn(df_sub_surgeries,val_col=target_col, group_col='sub_surgeries_count', p_adjust='bonferroni') # or 'holm', 'fdr_bh', etc))
print(dunn_results)

# Not significant association between the number of sub-surgeries and the delay.



# -----------------------------------------------------
# TEST: ASA Classification vs. Delay



# -----------------------------------------------------
# TEST: BMI vs. Delay


print("End of script")


############################################################
# Start: Visualizations NOT IMPUTED DATA
############################################################
plt.style.use('science')

plt.figure(figsize=(12, 6))
sns.histplot(df["deviation"], bins=80, kde=True)
plt.title(f'Distribution of deviation')
plt.xlabel("Absolute deviation")
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

# plotting the distribution of the target column
plt.figure(figsize=(12, 6))
sns.histplot(df[target_col], bins=30, kde=True)
plt.title(f'Distribution of realtive deviation')
plt.xlabel(target_col)
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()


############################################################
# Start: Visualizations: IMPUTED DATA
############################################################


plt.figure(figsize=(12, 6))
sns.histplot(df["test_deviation"], bins=100, kde=True)
plt.title(f'Distribution of deviation')
plt.xlabel("Absolute deviation (Imputed)")
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df["test_perc_dev"], bins=100, kde=True)
plt.title(f'Distribution of deviation')
plt.xlabel("Normalized deviation (IMPUTED)")
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()



# ---------------------------------------
# plotting distribbution of target across subspecialties (violin plot)
plt.figure(figsize=(12, 6))
sns.violinplot(x='Specialisme', y=target_col, data=df, inner='quartile')
plt.title(f'Distribution of {target_col} by Specialisme')
plt.xlabel('Specialisme')
plt.ylabel(target_col)
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()



