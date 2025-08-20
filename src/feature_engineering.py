
from typing import Tuple
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


from src.data_preparation import df_final

# -----------------------------
# Feature Engineering
# -----------------------------


def flag_outliers(df: pd.DataFrame, col: str):
    '''
    Instead of removing the outliers, flag outliers in a numeric column using IQR method.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the column to check.
    col : str
        Name of the numeric column to analyze for outliers.
    multiplier : float, optional
        The factor to multiply the IQR by to determine outlier bounds
        (default is 1.5).

    Returns
    -------
    pandas.DataFrame
        DataFrame with a new column 'Is Outlier' (1 if the row is an outlier, else 0).
    '''

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    # Determine outlier boundaries
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df['Is Outlier'] = (
        (df[col] < lower_bound) | (
            df[col] > upper_bound)).astype(int)

    return df


df_final = flag_outliers(df_final, 'Amount Paid')
df_final.loc[df_final['Is Outlier'] == 1].tail()  # Check the outliers rows

df = df_final.copy()


# -----------------------------
# Split into train/test
# -----------------------------
def features_train_test(df, target_col='Is Laundering'):
    '''Split to train and test set.'''

    # Split into train/test
    X = df.drop(columns=[target_col], axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# -----------------------------
# Encode and Scale variables
# -----------------------------
def features_transformation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
):
    '''
    Encode, scale, and apply SMOTE to training set.

    Parameters
    ----------
    x_train: pandas.DataFrame with training features.
    x_test: pandas.DataFrame with training features.
    y_train: pandas.Series with training target.
    y_test: pandas.Series with test target.

    Returns
    -------
    x_train_transformed: Transformed and SMOTE-resampled training features.
    y_train_resampled: Resampled training target.
    x_test_transformed: Transformed test features (no SMOTE applied).
    y_test: Original test target (unchanged).
    '''

    # Split features from target variables
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    num_cols = [col for col in df.columns if df[col].dtype != 'object' and col != 'Is Laundering']

    # Encode categorical variables + Scale numerical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', RobustScaler(), num_cols)
        ])

    # Apply preprocessing + SMOTE on train set
    X_train_encoded = preprocessor.fit_transform(X_train)
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_transformed, y_train_resampled = smote.fit_resample(
        X_train_encoded, y_train)  # type: ignore

    # Transform test
    X_test_transformed = preprocessor.transform(
        X_test)  # Transform test set - no SMOTE on test

    return X_train_transformed, y_train_resampled, X_test_transformed, y_test


X_train, X_test, y_train, y_test = features_train_test(df)
X_train, y_train, X_test, y_test = features_transformation(X_train, X_test, y_train, y_test)

# 9 785 635 each 'Is Laundering'

# -----------------------------
# Feature Importance and Selection
# -----------------------------

# Wrapper Methods: Recursive Feature Elimination using models that provide feature importance (tree-based or linear models)
# RFE
rf_rfe = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rfe_selector = RFE(rf_rfe, n_features_to_select=15, step=10)
rfe_selector.fit(X_train, y_train)
selected_features = X_train.columns[rfe_selector.support_]


#  Embedded Methods: feature selection is during the model training
# Feature Selection with Regularization -> L1 (Lasso) regularization (for numerical feature/linear models)
model = LogisticRegression(penalty='l1', solver='liblinear')
model.fit(X_train, y_train)
selected_features = X_train.columns[model.coef_[0] != 0]

# Tree-based feature importance (numerical and categorical features, non-linear relationships)
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
feature_ranking = pd.DataFrame(importances).sort_values(by='importance', ascending=False)


# -----------------------------
# Notes
# -----------------------------


# 'Timestamp', 'From Bank', 'Account Number', 'To Bank',
# 'Amount Received', 'Receiving Currency', 'Amount Paid',
# 'Payment Currency', 'Payment Format', 'Is Laundering', 'Bank Name',
# 'Entity ID', 'Entity Name', 'Is Outlier'


# 9 785 635 'Is Laundering'

# --------------------------
# Encode categorical variables: There is many approaches and it depends on the models that we plan to use afterwards
#   One-hot encoding -> increases dimensionality, SMOTE might create unrealistic combinations for high-cardinality features -> for distance-based or linear models (LogisticRegression, SVM, KNN)
#   Ordinal encoding -> imposes an artificial order that may mislead models that assume distance is meaningful -> for tree-based models because they don’t assume numeric distance meaning
#   Target/Mean Encoding -> replaces categories with statistics from the target
# Encode numerical variables
#   StandardScaler: uses mean and std. is sensible to
#   MinMaxScaler: uses min and max and scales the features in that range. Sensitive to outliers. Not robust to outliers. Preserves the original distribution shape
# RobustScaler: uses median and IQR instead of mean and std. less
# sensitive to outliers

# --------------------------
# As data is already splitted into train/test because while fitting an encoder on the entire dataset before information from the test set
# is “seen” during encoding which might lead to data leakage.
# Numerical data is right-skwed and it has outliers which decided not to
# remove them
