"""
Script Name: report_part1_regression.py
Author: Bennet Speh
Created: 2025-05-11
Description:
    This script is the final report version for the MUMC dataset. This script will give a an approach for the regression task.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.descriptivestats import describe
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as scs


# --------------------------------------------------
#                 CONFIGURATION
# --------------------------------------------------

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# target_col = 'absolute_perc_dev'
target_col = 'perc_dev'
# target_col = 'deviation'
csv_path = "/Users/bennet/Desktop/thesis/data/clean_MUMC_v3.csv"


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
df['deviation'] = df['Duur kamer'] - df['Booked']
# df['percentage_deviation'] = (df['Duur kamer'] - df['Booked']) / df['Booked'] * 100
df['has_delay'] = df['deviation'] > 10 # True if deviation is greater than 10%

# create columns overestimated and underestimated
df['overestimated'] = df['Duur kamer'] - df['Booked'] < -10
df['underestimated'] = df['Duur kamer'] - df['Booked'] > 10


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


# OUTPUT
print("Engineered data sample:")
print(df.head())
print(describe(df))
original_df = df.copy()


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
#                 OLS GLOBAL
# --------------------------------------------------
df = df.dropna(subset=['Time of day EXTENDED', 'Specialisme', 'Chirurg', target_col])
# X = df[['Time of day EXTENDED', 'Specialisme', 'Chirurg']]
X = df[['Time of day EXTENDED', 'Specialisme', 'Verrichtingen']]
X = pd.get_dummies(X, drop_first=True)

X = X.astype(int)
y = pd.to_numeric(df[target_col], errors='coerce')

X = X.dropna()
y = y[X.index]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# --------------------------------------------------
#        CHECK STATISTICAL ASSUMPTIONS FOR REGRESSION
# --------------------------------------------------
# 1. LINEARITY (scatter plot)
plt.figure(figsize=(8, 6))
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Linearity Check: Residuals vs Fitted")
plt.show()

# 2. MULTICOLLINEARITY (VIF)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\n📊 Variance Inflation Factors:")
print(vif_data)

# 3. NORMALITY OF RESIDUALS
residuals = model.resid
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title("Normality of Residuals")
plt.show()

sm.qqplot(residuals, line='45', fit=True)
plt.title("Q-Q Plot for Residuals")
plt.show()

shapiro_test = stats.shapiro(residuals)
print(f"\n🧪 Shapiro-Wilk Test for Normality: W={shapiro_test.statistic:.4f}, p={shapiro_test.pvalue:.4f}")

# 4. HOMOSCEDASTICITY
plt.figure(figsize=(8, 6))
plt.scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)))
plt.xlabel("Fitted Values")
plt.ylabel("Sqrt(Abs(Residuals))")
plt.title("Homoscedasticity Check")
plt.show()

# --------------------------------------------------
#         LASSO & RIDGE REGRESSION
# --------------------------------------------------

# Drop constant column before scikit-learn models
X_no_const = X.drop(columns=['const'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_no_const)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)
lasso_preds = lasso.predict(X_scaled)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
ridge_preds = ridge.predict(X_scaled)

# Model performance comparison
print("\n📈 Model Performance Comparison:")

print(f"\n🔹 LASSO: R^2 = {r2_score(y, lasso_preds):.4f}, MSE = {mean_squared_error(y, lasso_preds):.4f}")
print("Non-zero Lasso coefficients:")
print(pd.Series(lasso.coef_, index=X_no_const.columns)[lasso.coef_ != 0])

print(f"\n🔹 RIDGE: R^2 = {r2_score(y, ridge_preds):.4f}, MSE = {mean_squared_error(y, ridge_preds):.4f}")
print("Top Ridge coefficients:")
print(pd.Series(ridge.coef_, index=X_no_const.columns).sort_values(ascending=False).head(10))

# Optional: Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
lasso_cv_score = cross_val_score(lasso, X_scaled, y, cv=cv, scoring='r2')
ridge_cv_score = cross_val_score(ridge, X_scaled, y, cv=cv, scoring='r2')

print(f"\n🔁 Lasso CV mean R^2: {lasso_cv_score.mean():.4f}")
print(f"🔁 Ridge CV mean R^2: {ridge_cv_score.mean():.4f}")