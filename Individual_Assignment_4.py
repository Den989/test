# %% md
# # Credit Card Default Prediction
# ## ECON5130: Individual Assignment
# #
# ### Introduction
# #
# This notebook implements a comprehensive analysis of credit card default prediction using machine learning techniques. We compare three classification algorithms - Logistic Regression, Support Vector Machines (SVM), and Random Forest - to predict credit card defaults using data from the UCI Machine Learning Repository.
# #
# The analysis follows a structured approach:
# 1. Data Loading and Preprocessing
# 2. Exploratory Data Analysis
# 3. Feature Analysis and Selection
# 4. Feature Engineering
# 5. Model Development and Tuning
# 6. Performance Evaluation and Comparison
# 7. Business Impact Analysis
# #
# **Dataset Overview:**
# * Source: UCI Machine Learning Repository - "Default of Credit Card Clients"
# * Size: 30,000 credit card clients
# * Target Variable: Default payment next month (1 = Default, 0 = No Default)
# #
# %% md
# ### Setup and Required Libraries
# #
# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             precision_recall_curve, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# Set up the visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['axes.grid'] = True

# Custom color palette for consistency
colors = {
    'main': '#2E86C1',  # Primary blue
    'secondary': '#E74C3C',  # Red for contrast
    'highlight': '#27AE60',  # Green for highlights
    'neutral': '#7F8C8D',  # Gray for background
    'rf': '#8E44AD'  # Purple for Random Forest
}

# %% md
# ### Data Loading and Initial Processing
# #
# First, we'll load and preprocess the credit card default dataset.
# #
# %%
# Load the dataset
# Clean up column names
column_names = {
    'X1': 'LIMIT_BAL',
    'X2': 'SEX',
    'X3': 'EDUCATION',
    'X4': 'MARRIAGE',
    'X5': 'AGE',
    'X6': 'PAY_0',
    'X7': 'PAY_2',
    'X8': 'PAY_3',
    'X9': 'PAY_4',
    'X10': 'PAY_5',
    'X11': 'PAY_6',
    'X12': 'BILL_AMT1',
    'X13': 'BILL_AMT2',
    'X14': 'BILL_AMT3',
    'X15': 'BILL_AMT4',
    'X16': 'BILL_AMT5',
    'X17': 'BILL_AMT6',
    'X18': 'PAY_AMT1',
    'X19': 'PAY_AMT2',
    'X20': 'PAY_AMT3',
    'X21': 'PAY_AMT4',
    'X22': 'PAY_AMT5',
    'X23': 'PAY_AMT6',
    'Y': 'DEFAULT'
}

df = pd.read_csv('credit_card_defaults.csv')
df = df.rename(columns=column_names)
df = df.drop(df.columns[0], axis=1) if 'Unnamed: 0' in df.columns else df  # Drop the unnamed index column

# Convert all columns to appropriate numeric types
numeric_columns = df.columns
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values with median
df = df.fillna(df.median())

# Display basic information about the dataset
print("Dataset Overview:")
print(f"Number of samples: {len(df):,}")
print(f"Number of features: {df.shape[1] - 1}")  # Excluding target variable
print(f"\nDefault rate: {(df['DEFAULT'].mean() * 100):.1f}%")  # Shows 22.1% default rate

# %% md
# ### Exploratory Data Analysis
# #
# Now let's explore the dataset to understand the distribution of the target variable, demographic patterns, and key characteristics.
# #
# %%
# Create a figure for categorical variables distribution
plt.figure(figsize=(20, 12))

# Plot 1: Default Distribution with percentages
plt.subplot(2, 2, 1)
default_counts = df['DEFAULT'].value_counts().sort_index()
ax1 = sns.barplot(x=default_counts.index, y=default_counts.values, color=colors['main'], edgecolor='black')
plt.title('Distribution of Default vs Non-Default', fontsize=14)
plt.xlabel('Default Status (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, alpha=0.3)
# Add percentage labels
total = len(df)
for i, count in enumerate(default_counts.values):
    percentage = count / total * 100
    plt.text(i, count + 200, f'{percentage:.1f}%', ha='center', fontsize=11)

# Plot 2: Education Distribution with percentages
plt.subplot(2, 2, 2)
# Create clean x-axis labels and filter valid education values (1-4)
edu_labels = {
    1: 'Graduate',
    2: 'University',
    3: 'High School',
    4: 'Others'
}
# Filter to include only valid education codes
valid_edu_codes = [1, 2, 3, 4]
valid_education = df[df['EDUCATION'].isin(valid_edu_codes)]['EDUCATION'].value_counts().sort_index()
ax2 = sns.barplot(x=valid_education.index, y=valid_education.values, color=colors['secondary'], edgecolor='black')
plt.title('Distribution of Education Levels', fontsize=14)
plt.xlabel('Education Level', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(len(valid_education.index)),
           [edu_labels.get(idx, 'Unknown') for idx in valid_education.index])

# Add percentage labels
valid_total = valid_education.sum()  # Use filtered data for percentage calculation
for i, count in enumerate(valid_education.values):
    percentage = count / valid_total * 100
    plt.text(i, count + 200, f'{percentage:.1f}%', ha='center', fontsize=11)

# Plot 3: Marriage Distribution with percentages
plt.subplot(2, 2, 3)
# Create clean x-axis labels and filter valid marriage values (1-3)
marriage_labels = {
    1: 'Married',
    2: 'Single',
    3: 'Others'
}
# Filter to include only valid marriage codes
valid_marriage_codes = [1, 2, 3]
valid_marriage = df[df['MARRIAGE'].isin(valid_marriage_codes)]['MARRIAGE'].value_counts().sort_index()
ax3 = sns.barplot(x=valid_marriage.index, y=valid_marriage.values, color=colors['highlight'], edgecolor='black')
plt.title('Distribution of Marital Status', fontsize=14)
plt.xlabel('Marital Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(len(valid_marriage.index)),
           [marriage_labels.get(idx, 'Unknown') for idx in valid_marriage.index])

# Add percentage labels
valid_total_marriage = valid_marriage.sum()  # Use filtered data for percentage calculation
for i, count in enumerate(valid_marriage.values):
    percentage = count / valid_total_marriage * 100
    plt.text(i, count + 200, f'{percentage:.1f}%', ha='center', fontsize=11)

# Plot 4: Age Distribution (no percentages needed for continuous variable)
plt.subplot(2, 2, 4)
sns.histplot(data=df, x='AGE', bins=40, color=colors['main'], kde=True, edgecolor='black')
plt.title('Age Distribution', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)
plt.show()

# %% md
# ### Interpretation of Demographic Distribution Graphs
# #
# The demographic distribution graphs reveal important patterns in our dataset:
# #
# - **Default Distribution**: Approximately 22.1% of clients defaulted, showing this is an imbalanced classification problem.
# #
# - **Education Levels**: University graduates represent the largest segment, followed by high school graduates. This suggests our model needs to work effectively across different educational backgrounds.
# #
# - **Marital Status**: Married individuals form the largest group, with single clients as the second largest category. The "Others" category is relatively small.
# #
# - **Age Distribution**: Client ages follow a roughly normal distribution centered around 35-40 years, with fewer very young or elderly clients.
# #
# These demographic insights help contextualize our modeling approach and potential biases in prediction.
# #
# %% md
# ## Feature Analysis and Correlation Studies
# #
# We analyze relationships between variables to:
# 1. Identify features most strongly correlated with default risk
# 2. Detect potential multicollinearity
# 3. Understand payment history patterns
# 4. Guide feature selection for modeling
# #
# This analysis will inform our feature selection and help us understand which variables are most predictive of credit card defaults.
# #
# %%
# Compute the correlation matrix
correlation_matrix = df.corr()

# Visualize the correlation matrix
plt.figure(figsize=(16, 12))
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=False, mask=mask, cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.grid(False)  # No grid for heatmap
plt.tight_layout()
plt.show()

# Calculate absolute correlations with DEFAULT
default_correlations_abs = correlation_matrix['DEFAULT'].abs().sort_values(ascending=False)

# Calculate the total correlation (for percentage)
total_correlation = default_correlations_abs[1:11].sum()  # Exclude DEFAULT's correlation with itself

# Print top 10 features correlated with default risk with headers and percentages
print("\nTop 10 Features Correlated with Default Risk:")
print(f"{'Feature':<15} {'Correlation':<12} {'Percentage':<10}")
print("-" * 40)
for feature in default_correlations_abs[1:11].index:
    corr_value = default_correlations_abs[feature]
    percentage = (corr_value / total_correlation * 100)
    print(f"{feature:<15} {corr_value:.4f}      {percentage:.1f}%")

# Examine correlation between payment history variables
payment_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
payment_corr = correlation_matrix.loc[payment_cols, payment_cols]
plt.figure(figsize=(10, 8))
sns.heatmap(payment_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
plt.title('Correlation Between Payment History Variables', fontsize=14)
plt.grid(False)  # No grid for heatmap
plt.tight_layout()
plt.show()

# %% md
# ### Key Findings from Correlation Analysis:
# #
# 1. **Payment History Impact:**
#    - Recent payment history (PAY_0, PAY_2) shows the strongest correlation with default risk
#    - The impact of payment history diminishes for older months
# #
# 2. **Financial Indicators:**
#    - Bill amounts show moderate correlation with default probability
#    - Credit limit (LIMIT_BAL) has a negative correlation, suggesting higher limits may indicate lower risk
# #
# 3. **Demographic Factors:**
#    - Demographic variables (SEX, EDUCATION, MARRIAGE) show relatively weak correlation with default
#    - Age has a minimal correlation with default probability
# #
# 4. **Feature Selection Implications:**
#    - Payment history variables will be crucial predictors
#    - Recent payment history may be more valuable than older history
#    - Bill amounts and credit limit should be retained as important features
#    - Demographic variables might add limited predictive value
# #
# %% md
# ### Payment Pattern Analysis
# #
# Payment history is a critical factor in predicting defaults. Let's examine:
# - How payment behavior relates to default rates
# - Patterns across different payment periods
# - The predictive power of payment status variables
# #
# %%
# Visualize the relationship between payment history and default
plt.figure(figsize=(20, 12))
for i, col in enumerate(payment_cols):
    plt.subplot(2, 3, i + 1)

    # Calculate default rate for each payment status
    default_rates = df.groupby(col)['DEFAULT'].mean() * 100

    # Create bar plot
    ax = sns.barplot(x=default_rates.index, y=default_rates.values, color=colors['main'], edgecolor='black')
    plt.title(f'Default Rate by {col} Status', fontsize=12)
    plt.xlabel('Payment Status', fontsize=10)
    plt.ylabel('Default Rate (%)', fontsize=10)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)

    # Add value labels
    for j, rate in enumerate(default_rates):
        plt.text(j, rate + 2, f'{rate:.1f}%', ha='center')

plt.tight_layout(pad=3.0)
plt.show()

# %% md
# ## Insights from Correlation Analysis:
# #
# 1. Payment history variables (PAY_0 to PAY_6) show the strongest correlation with default risk.
# 2. There's high correlation among consecutive payment history months, suggesting potential multicollinearity.
# 3. Bill amounts show moderate correlation with default status.
# 4. Demographic variables (SEX, EDUCATION, MARRIAGE) show relatively weak correlation with default.
# 5. Credit limit (LIMIT_BAL) has a negative correlation with default, suggesting higher limits may be associated with lower default risk.
# #
#
# %% md
# ## Enhanced Demographic Analysis
# #
# Let's examine how default rates vary across demographic factors and credit attributes.
# This will provide insights into the risk factors associated with different customer segments.
#
# %%
# Create a separate DataFrame for visualization only
df_viz = df.copy()

# 1. Default rate by Age Group
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Use integer-based bins for age groups
age_bins = [20, 30, 40, 50, 60, 100]
age_labels = ['20-30', '31-40', '41-50', '51-60', '60+']
df_viz['AGE_GROUP'] = pd.cut(df_viz['AGE'], bins=age_bins, labels=age_labels)

# Clean the education data - remove unknown or invalid values
df_viz = df_viz[df_viz['EDUCATION'].isin([1, 2, 3, 4])]

# Plot default rate by age group
age_default = df_viz.groupby('AGE_GROUP')['DEFAULT'].mean() * 100
sns.barplot(x=age_default.index, y=age_default.values, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Default Rate by Age Group', fontsize=14)
axes[0, 0].set_xlabel('Age Group', fontsize=12)
axes[0, 0].set_ylabel('Default Rate (%)', fontsize=12)
axes[0, 0].grid(axis='y', alpha=0.3)
# Add percentage labels
for i, rate in enumerate(age_default):
    axes[0, 0].text(i, rate + 1, f'{rate:.1f}%', ha='center')

# 2. Default rate by Education
valid_education = [1, 2, 3, 4]  # Only include valid education codes
education_default = df_viz[df_viz['EDUCATION'].isin(valid_education)].groupby('EDUCATION')['DEFAULT'].mean() * 100
# Map education codes to meaningful names
education_map = {1: 'Graduate', 2: 'University', 3: 'High School', 4: 'Others'}
education_labels = [education_map.get(i, str(i)) for i in education_default.index]

sns.barplot(x=education_labels, y=education_default.values, ax=axes[0, 1], palette='plasma')
axes[0, 1].set_title('Default Rate by Education Level', fontsize=14)
axes[0, 1].set_xlabel('Education Level', fontsize=12)
axes[0, 1].set_ylabel('Default Rate (%)', fontsize=12)
axes[0, 1].grid(axis='y', alpha=0.3)
# Add percentage labels
for i, rate in enumerate(education_default):
    axes[0, 1].text(i, rate + 1, f'{rate:.1f}%', ha='center')

# 3. Default rate by Marriage Status
valid_marriage = [1, 2, 3]  # Only include valid marriage codes
marriage_default = df_viz[df_viz['MARRIAGE'].isin(valid_marriage)].groupby('MARRIAGE')['DEFAULT'].mean() * 100
# Map marriage codes to meaningful names
marriage_map = {1: 'Married', 2: 'Single', 3: 'Others'}
marriage_labels = [marriage_map.get(i, str(i)) for i in marriage_default.index]

sns.barplot(x=marriage_labels, y=marriage_default.values, ax=axes[1, 0], palette='crest')
axes[1, 0].set_title('Default Rate by Marital Status', fontsize=14)
axes[1, 0].set_xlabel('Marital Status', fontsize=12)
axes[1, 0].set_ylabel('Default Rate (%)', fontsize=12)
axes[1, 0].grid(axis='y', alpha=0.3)
# Add percentage labels
for i, rate in enumerate(marriage_default):
    axes[1, 0].text(i, rate + 1, f'{rate:.1f}%', ha='center')

# 4. Default rate by Credit Limit Range (using integers for bins)
limit_bins = [0, 50000, 100000, 200000, 300000, 1000000]
limit_labels = ['<50K', '50K-100K', '100K-200K', '200K-300K', '>300K']
df_viz['LIMIT_RANGE'] = pd.cut(df_viz['LIMIT_BAL'], bins=limit_bins, labels=limit_labels)

limit_default = df_viz.groupby('LIMIT_RANGE')['DEFAULT'].mean() * 100
sns.barplot(x=limit_default.index, y=limit_default.values, ax=axes[1, 1], palette='magma')
axes[1, 1].set_title('Default Rate by Credit Limit Range', fontsize=14)
axes[1, 1].set_xlabel('Credit Limit Range (NT$)', fontsize=12)
axes[1, 1].set_ylabel('Default Rate (%)', fontsize=12)
axes[1, 1].grid(axis='y', alpha=0.3)
# Add percentage labels
for i, rate in enumerate(limit_default):
    axes[1, 1].text(i, rate + 1, f'{rate:.1f}%', ha='center')

plt.tight_layout()
plt.show()

# Important: Don't store these categorical variables in the main dataframe
del df_viz  # Delete the visualization dataframe when done

# %% md
# ## Demographic Analysis Insights
# #
# Our enhanced demographic analysis reveals several important risk factors:
# #
# 1. **Age Impact**: Younger clients (20-30) show higher default rates, while middle-aged clients (40-50) demonstrate lower risk.
# #
# 2. **Education Correlation**: Higher education levels generally correlate with lower default rates, with graduate degree holders showing the lowest risk.
# #
# 3. **Marital Status Significance**: Married individuals tend to have lower default rates compared to single individuals, possibly reflecting greater financial stability.
# #
# 4. **Credit Limit as Risk Indicator**: Lower credit limits (<50K) correlate with significantly higher default rates, likely reflecting the bank's prior risk assessment.
# #
# These demographic patterns provide valuable context for our modeling approach, suggesting that age, education, marital status, and approved credit limits all serve as useful signals for default risk.
#
# %% md
# ## Feature Engineering
# #
# Let's create new features that might improve our predictive power:
# 1. Credit utilization ratios
# 2. Payment to bill ratios
# 3. Payment trends over time
# 4. Delinquency indicators
# #
# %%
# Create a copy of the original dataframe for feature engineering
df_fe = df.copy()

# 1. Credit Utilization Ratios
print("Creating utilization ratios...")
for i in range(1, 7):
    # Create credit utilization ratio (bill amount / credit limit)
    # Handle zero division by replacing with 0 (no utilization)
    col_name = f'UTIL_RATIO_{i}'
    df_fe[col_name] = df_fe[f'BILL_AMT{i}'] / df_fe['LIMIT_BAL']
    df_fe[col_name] = df_fe[col_name].replace([np.inf, -np.inf], np.nan).fillna(0)
    # Cap at 100% utilization for interpretation (some values might exceed 1)
    df_fe[col_name] = df_fe[col_name].clip(upper=1.0)

    # Print summary statistics
    print(f"Utilization Ratio {i} - Mean: {df_fe[col_name].mean():.4f}, Max: {df_fe[col_name].max():.4f}")

# 2. Payment to Bill Ratios
print("\nCreating payment ratios...")
for i in range(1, 7):
    # Create payment ratio (payment amount / bill amount)
    col_name = f'PAY_RATIO_{i}'
    # Avoid division by zero by handling small or zero bill amounts
    df_fe[col_name] = np.where(
        df_fe[f'BILL_AMT{i}'] > 100,  # Only calculate ratio for meaningful bills
        df_fe[f'PAY_AMT{i}'] / df_fe[f'BILL_AMT{i}'],
        1.0  # Assume full payment for very small bills
    )
    df_fe[col_name] = df_fe[col_name].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    # Cap at 100% payment ratio for interpretation
    df_fe[col_name] = df_fe[col_name].clip(upper=1.0)

    # Print summary statistics
    print(f"Payment Ratio {i} - Mean: {df_fe[col_name].mean():.4f}, Max: {df_fe[col_name].max():.4f}")

# 3. Payment Trend Indicators
print("\nCreating payment trend indicators...")
# Calculate the 3-month average utilization
df_fe['AVG_UTIL_3M'] = df_fe[['UTIL_RATIO_1', 'UTIL_RATIO_2', 'UTIL_RATIO_3']].mean(axis=1)
# Calculate the 6-month average utilization
df_fe['AVG_UTIL_6M'] = df_fe[['UTIL_RATIO_1', 'UTIL_RATIO_2', 'UTIL_RATIO_3',
                              'UTIL_RATIO_4', 'UTIL_RATIO_5', 'UTIL_RATIO_6']].mean(axis=1)
# Calculate utilization trend (recent 3 months vs previous 3 months)
recent_util = df_fe[['UTIL_RATIO_1', 'UTIL_RATIO_2', 'UTIL_RATIO_3']].mean(axis=1)
older_util = df_fe[['UTIL_RATIO_4', 'UTIL_RATIO_5', 'UTIL_RATIO_6']].mean(axis=1)
df_fe['UTIL_TREND'] = recent_util - older_util
# Handling cases where older_util is 0
df_fe['UTIL_TREND'] = df_fe['UTIL_TREND'].replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Utilization Trend - Mean: {df_fe['UTIL_TREND'].mean():.4f}, "
      f"Min: {df_fe['UTIL_TREND'].min():.4f}, Max: {df_fe['UTIL_TREND'].max():.4f}")

# 4. Delinquency Indicators
print("\nCreating delinquency indicators...")
# Count number of months with any delay (PAY_x > 0)
delinquency_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
df_fe['DELINQUENCY_COUNT'] = (df_fe[delinquency_cols] > 0).sum(axis=1)
# Calculate average delinquency level across all months
df_fe['AVG_DELINQUENCY'] = df_fe[delinquency_cols].clip(lower=0).mean(axis=1)
# Create binary indicator for serious delinquency (2+ months delay)
df_fe['SERIOUS_DELINQ'] = (df_fe[delinquency_cols] >= 2).any(axis=1).astype(int)

print(f"Delinquency Count - Mean: {df_fe['DELINQUENCY_COUNT'].mean():.2f}, "
      f"Max: {df_fe['DELINQUENCY_COUNT'].max():.2f}")
print(f"Average Delinquency - Mean: {df_fe['AVG_DELINQUENCY'].mean():.2f}, "
      f"Max: {df_fe['AVG_DELINQUENCY'].max():.2f}")
print(f"Serious Delinquency Rate: {df_fe['SERIOUS_DELINQ'].mean() * 100:.2f}%")

# 5. Additional demographic features
# Create age groups
age_bins = [20, 30, 40, 50, 60, 100]
age_labels = ['20-30', '31-40', '41-50', '51-60', '60+']
df_fe['AGE_GROUP'] = pd.cut(df_fe['AGE'], bins=age_bins, labels=age_labels)
# Convert to one-hot encoding
age_dummies = pd.get_dummies(df_fe['AGE_GROUP'], prefix='AGE_GROUP')
df_fe = pd.concat([df_fe, age_dummies], axis=1)
# Drop the categorical column
df_fe.drop('AGE_GROUP', axis=1, inplace=True)

# Print the final engineered feature list
new_features = [col for col in df_fe.columns if col not in df.columns]
print(f"\nAdded {len(new_features)} new engineered features:")
for feature in new_features:
    print(f"- {feature}")

# Visualize the correlation of new features with DEFAULT
plt.figure(figsize=(14, 10))
# Select top 15 engineered features by correlation with DEFAULT
new_features_corr = df_fe[new_features + ['DEFAULT']].corr()['DEFAULT'].abs().sort_values(ascending=False)
top_features = new_features_corr.head(15).index.tolist()
# Remove DEFAULT from the list
if 'DEFAULT' in top_features:
    top_features.remove('DEFAULT')

# Create correlation heatmap
top_features_corr = df_fe[top_features + ['DEFAULT']].corr()
sns.heatmap(top_features_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
plt.title('Correlation Between New Features and Default', fontsize=14)
plt.tight_layout()
plt.show()

# Show correlation of top 10 engineered features with DEFAULT
print("\nTop 10 Engineered Features by Correlation with DEFAULT:")
print(f"{'Feature':<20} {'Correlation':<12}")
print("-" * 35)
for feature in new_features_corr.head(10).index:
    if feature != 'DEFAULT':
        corr_value = new_features_corr[feature]
        print(f"{feature:<20} {corr_value:.4f}")

# %% md
# ## Data Preparation
# #
# We prepare the data for modeling by:
# 1. Handling any missing values
# 2. Scaling features
# 3. Splitting into training and testing sets
# #
# %%
# Use the engineered feature dataframe from now on
df = df_fe

# Make sure all columns are numeric before splitting
X = df.drop(['DEFAULT'], axis=1)
y = df['DEFAULT'].astype(int)

# Print data check
print("X shape:", X.shape)
print("y shape:", y.shape)

# Check for non-numeric columns that might cause issues
non_numeric_cols = []
for col in X.columns:
    if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.CategoricalDtype):
        non_numeric_cols.append(col)
        print(f"Found non-numeric column: {col}")

if non_numeric_cols:
    print(f"Removing {len(non_numeric_cols)} non-numeric columns from feature set")
    X = X.drop(columns=non_numeric_cols)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Print the shape of the training and testing sets
print("Dataset Splitting:")
print(f"Training set shape: {X_train.shape} samples, {X_train.shape[1]} features")