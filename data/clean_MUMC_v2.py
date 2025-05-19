"""
This script is used to extract, clean, and engineer features from the MUMC OR log dataset.
It includes decryption of an Excel file, filtering relevant rooms, formatting datetime entries,
computing derived metrics (e.g., booking duration), and preparing the dataset for analysis.
"""

import msoffcrypto
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# FILE_PATH = "/Users/bennet/Desktop/thesis/data/original_MUMC_log.xlsx"
FILE_PATH = "/Users/bennet/Downloads/Bennet_20250519_0753.xlsx"
FILE_PASSWORD = "ok"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# ------------------------------------------------------------------------------
# Data Loading and Decryption
# ------------------------------------------------------------------------------

def load_decrypted_excel(file_path, password):
    """
    Decrypts and loads a password-protected Excel file.
    """
    decrypted = BytesIO()
    with open(file_path, "rb") as f:
        office_file = msoffcrypto.OfficeFile(f)
        office_file.load_key(password=password)
        office_file.decrypt(decrypted)
    decrypted.seek(0)
    return pd.read_excel(decrypted)


df = load_decrypted_excel(FILE_PATH, FILE_PASSWORD)

print("Original data sample:")
print(df.head())

# ------------------------------------------------------------------------------
# Filtering Relevant Rooms
# ------------------------------------------------------------------------------

# Filter for operating rooms of interest
df = df[df["Kamer"].str.contains("OC OK|OC VH", regex=True)]

# Select key columns
"""
selected_columns = [
    "Datum", "Chirurg", "Verrichtingen", "Toevoeging", "In kamer", "Uit kamer", "Duur kamer",
    "Patiëntklasse", "Specialisme", "Verantw. anest.zorgverl.", "Primair anest.type", "Kamer",
    "Verrichtingsniveau logboek", "Logboeknr.", "Patiëntleeftijd", "Medewerkers", "Dag vd week",
    "In rapportering org.?", "Minuten afwijkend", "Minuten onderplanning", "Minuten overplanning", "Vertragingsreden"
]
"""
selected_columns = ["Datum", "Dag vd week", "Naam patiënt", "Hoofdchirurg/zorgverlener", "Specialisme", "Kamer", "Verrichtingen", "In kamer", "Start verr.", "Verr. afger.", "Uit kamer" ,"Duur kamer", "Duur verr.", "Patiëntleeftijd op verrichtingsdatum", "Patiëntklasse", "Minuten overplanning" ,"Minuten onderplanning", "Primair anest.type"]

df = df[selected_columns].copy()

# ------------------------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------------------------

# Convert time columns
df["In kamer"] = pd.to_datetime(df["In kamer"], format="%H:%M:%S", errors="coerce").dt.time
df["Uit kamer"] = pd.to_datetime(df["Uit kamer"], format="%H:%M:%S", errors="coerce").dt.time

# Fill NaNs in planning deviation columns
df["Minuten onderplanning"] = df["Minuten onderplanning"].fillna(0)
df["Minuten overplanning"] = df["Minuten overplanning"].fillna(0)


# --- Booked Duration ---
def calculate_booked(row):
    if row["Duur kamer"] > 0:
        return row["Duur kamer"] + row["Minuten overplanning"] - row["Minuten onderplanning"]
    else:
        return row["Duur kamer"] # - row["Minuten afwijkend"]


df["Booked"] = df.apply(calculate_booked, axis=1)


# --- Time of Day Classification ---
def time_of_day(row):
    if row["In kamer"] and pd.to_datetime("06:00:00").time() <= row["In kamer"] < pd.to_datetime("12:00:00").time():
        return "Morning"
    return "Afternoon"


df["Time of day"] = df.apply(time_of_day, axis=1)

# --- Historical Average Duration ---
df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y", errors="coerce")
df = df.sort_values(["Verrichtingen", "Datum"])
df["AVG_DUR_5_PREV_SUR"] = df.groupby("Verrichtingen")["Duur kamer"].transform(
    lambda x: x.shift().rolling(5, min_periods=1).mean()
)
df["AVG_DUR_5_PREV_SUR"] = df["AVG_DUR_5_PREV_SUR"].fillna(df["Duur kamer"])


# ------------------------------------------------------------------------------
# Patient Age Conversion
# ------------------------------------------------------------------------------

def convert_to_years(age):
    if isinstance(age, str):
        if "jr" in age:
            return float(age.split()[0])
        elif "mnd" in age:
            return float(age.split()[0]) / 12
        elif "wkn" in age:
            return float(age.split()[0]) / 52
    return None


df["Patiëntleeftijd op verrichtingsdatum"] = df["Patiëntleeftijd op verrichtingsdatum"].apply(convert_to_years)

# ------------------------------------------------------------------------------
# Filtering and Final Cleanup
# ------------------------------------------------------------------------------

# Drop rows with missing durations
df_filtered = df.dropna(subset=["Duur kamer"])

# Sort by date
df_filtered = df_filtered.sort_values(by=["Datum"])


# ------------------------------------------------------------------------------
# NUMBER OF DAILY SURGERIES LED BY SURGEON
# ------------------------------------------------------------------------------
def count_daily_surgeries(df):
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y", errors="coerce")
    surgeries_per_day = df.groupby(["Datum", "Hoofdchirurg/zorgverlener"]).size().reset_index(name="Number of Surgeries per Surgent")
    return surgeries_per_day


# Count daily surgeries
daily_surgeries = count_daily_surgeries(df_filtered)

# Merge the daily surgery count into the main DataFrame
df_filtered = df_filtered.merge(daily_surgeries, on=["Datum", "Hoofdchirurg/zorgverlener"], how="left")


def count_surgeries_per_specialism(df):
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y", errors="coerce")
    surgeries_per_specialism = df.groupby(["Datum", "Specialisme"]).size().reset_index(name="Number of Surgeries per Specialism per day")
    return surgeries_per_specialism


# Count surgeries per specialism
specialism_surgeries = count_surgeries_per_specialism(df_filtered)
# Merge the specialism surgery count into the main DataFrame
df_filtered = df_filtered.merge(specialism_surgeries, on=["Datum", "Specialisme"], how="left")


def count_total_surgeries_per_day(df):
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y", errors="coerce")
    total_surgeries_per_day = df.groupby("Datum").size().reset_index(name="Total Surgeries per Day")
    return total_surgeries_per_day

# Count total surgeries per day
total_surgeries = count_total_surgeries_per_day(df_filtered)
# Merge the total surgery count into the main DataFrame
df_filtered = df_filtered.merge(total_surgeries, on="Datum", how="left")


def count_surgeries_last_day_and_week(df):
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y", errors="coerce")

    # Ensure data is sorted for performance (optional, but good practice)
    df = df.sort_values(by="Datum")

    def count_prev_day_and_week(row):
        surgeon = row["Hoofdchirurg/zorgverlener"]
        date = row["Datum"]

        # Previous day range
        prev_day = date - pd.Timedelta(days=1)
        same_surgeon = df["Hoofdchirurg/zorgverlener"] == surgeon

        prev_day_surgeries = df[(same_surgeon) & (df["Datum"].dt.date == prev_day.date())]
        prev_7d_surgeries = df[(same_surgeon) & (df["Datum"] >= (date - pd.Timedelta(days=7))) & (df["Datum"] < date)]

        return pd.Series({
            "Surgeries Previous Day": prev_day_surgeries.shape[0],
            "Surgeries Last 7 Days (excluding today)": prev_7d_surgeries.shape[0]
        })

    df[["Surgeries Previous Day", "Surgeries Last 7 Days (excluding today)"]] = df.apply(count_prev_day_and_week, axis=1)
    return df

# Count surgeries in the last day and week
df_filtered = count_surgeries_last_day_and_week(df_filtered)


# ------------------------------------------------------------------------------
# More filters
# ------------------------------------------------------------------------------
df_filtered = df_filtered[~df_filtered['Dag vd week'].isin(['za', 'zo'])]
# Filter out shift times (shift is from 07:00 to 17:30) start time
# Convert "In kamer" to datetime, then extract time
df_filtered["In kamer"] = pd.to_datetime(df_filtered["In kamer"], format="%H:%M:%S", errors='coerce').dt.time

# Define shift start and end times
shift_start = pd.to_datetime("07:00:00", format="%H:%M:%S").time()
shift_end = pd.to_datetime("17:30:00", format="%H:%M:%S").time()
print("Before filtering for shift:", df_filtered.shape[0], "rows")
# Filter rows based on shift times
df_filtered = df_filtered[df_filtered["In kamer"].between(shift_start, shift_end)]
print("After filtering for shift:", df_filtered.shape[0], "rows")





# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------
"""
# --- Boxplot of Durations by Specialism ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_filtered, x="Specialisme", y="Duur kamer")
plt.xticks(rotation=90)
plt.title("Duration of Surgeries per Specialism")
plt.ylabel("Duration (minutes)")
plt.tight_layout()
plt.show()

# --- Operation Counts per Specialism ---
plt.figure(figsize=(12, 6))
sns.countplot(data=df_filtered, x="Specialisme")
plt.xticks(rotation=90)
plt.title("Number of Operations per Specialism")
plt.ylabel("Number of Operations")
plt.tight_layout()
plt.show()
"""
# Add numbers of previous surgeries per patient and number of days since last surgery
def running_prev_surgeries_per_patient(df):
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y", errors="coerce")
    df = df.sort_values(by=["Naam patiënt", "Datum"])
    df["Previous Surgeries"] = df.groupby("Naam patiënt").cumcount()
    return df


def days_since_last_surgery(df):
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y", errors="coerce")
    df["Days Since Last Surgery"] = (df["Datum"] - df.groupby("Naam patiënt")["Datum"].shift()).dt.days
    return df

# Apply the functions to create new columns
df_filtered = running_prev_surgeries_per_patient(df_filtered)
df_filtered = days_since_last_surgery(df_filtered)

# sort by date
df_filtered = df_filtered.sort_values(by=["Datum"])


# ------------------------------------------------------------------------------
# Output and Inspection
# ------------------------------------------------------------------------------

print("Filtered data sample:")
print(df_filtered.head())

print("\nSummary statistics:")
print(df_filtered.describe())

# Optional: Export cleaned data
df_filtered.to_csv("/Users/bennet/Desktop/thesis/data/clean_MUMC_v4.csv", index=False)
