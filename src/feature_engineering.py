from typing import Tuple, List
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

from src.data_preparation import df_final

# -------------------
# Feature Engineering
# -------------------


def flag_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    '''
    Instead of removing the outliers, flag outliers in a numeric column using IQR method.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the column to check.
    col : str
        Name of the numeric column to analyze for outliers.

    Returns
    -------
    pd.DataFrame with a new column 'Is Outlier' (1 if the row is an outlier, else 0).
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

def features_train_test(df: pd.DataFrame,
                        target_col: str = 'Is Laundering') -> Tuple[pd.DataFrame,
                                                                    pd.DataFrame,
                                                                    pd.Series,
                                                                    pd.Series]:
    '''
    Splits dataframe into train and test sets.

    Parameters
    ----------
    pd.DataFrame: Input dataset including features and target.
    target_col : str, optional
        Name of the target column (default is 'Is Laundering').

    Returns
    -------
    X_train : Training features.
    X_test : Test features.
    y_train : Training target.
    y_test : Test target.
    '''

    # Split into train/test
    X = df.drop(columns=[target_col], axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# -------------------------
# Encode and Scale features
# -------------------------

def features_transformation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple:
    '''
    Encode, scale, and apply SMOTE to training set.

    Parameters
    ----------
    X_train: pandas.DataFrame with training features.
    X_test: pandas.DataFrame with training features.

    Returns
    -------
    X_train_transformed: Transformed training features.
    X_test_transformed: Transformed test features.
    '''

    # Split features from target variables
    cat_cols = [
        col for col in X_train.columns if X_train[col].dtype == 'object']
    num_cols = [col for col in X_train.columns if X_train[col].dtype !=
                'object' and col != 'Is Laundering']

    # Encode categorical variables + Scale numerical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', RobustScaler(), num_cols)
        ])

    # Apply preprocessing on train and test set
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    return X_train_encoded, X_test_encoded


# --------------------------------
# Feature Importance and Selection
# --------------------------------

# Wrapper Methods: Recursive Feature Elimination using models that provide
# feature importance (tree-based or linear models)
def rfe_feature_selection(
        X_train: pd.DataFrame,
        y_train: pd.Series) -> List[str]:
    '''
    Recursive Feature Elimination (RFE) using Random Forest.

    Parameters
    ----------
    X : pd.DataFrame Feature matrix.
    y : pd.Series Target vector.

    Returns
    -------
    List[str] of selected feature names.
    '''

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rfe_selector = RFE(rf, n_features_to_select=15, step=10)
    rfe_selector.fit(X_train, y_train)

    return X_train.columns[rfe_selector.support_].tolist()


# Embedded Methods: feature selection is during the model training
def l1_feature_selection(
        X_train: pd.DataFrame,
        y_train: pd.Series) -> List[str]:
    '''
    Perform feature selection using L1-regularization (LASSO) Logistic Regression
    for numerical feature/linear models.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    List[str] of selected feature names.
    '''
    lr = LogisticRegression(penalty='l1', solver='liblinear')
    lr.fit(X_train, y_train)

    return X_train.columns[lr.coef_[0] != 0].tolist()


def tree_feature_importance(
        X_train: pd.DataFrame,
        y_train: pd.Series) -> pd.DataFrame:
    '''
    Compute feature importance using Random Forest. A Tree-based feature
    importance (numerical and categorical features, non-linear relationships).

    Parameters
    ----------
    X : pd.DataFrame Feature matrix.
    y : pd.Series Target vector.

    Returns
    -------
    pd.DataFrame with 'feature' and 'importance' columns sorted descending.
    '''
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

    return pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)


# -----------
# Apply SMOTE
# -----------

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    '''
    Resample the training set using SMOTE on selected features.
    No SMOTE applied on test.

    Parameters
    ----------
    X_train : pd.DataFrame Training feature matrix.
    y_train : pd.Series Training target vector.
    selected_features : List[str]
        List of feature names to apply SMOTE on.

    Returns
    -------
    X_train_transformed : Resampled training features.
    y_train_resampled : Resampled training target.
    '''

    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_selected = X_train[selected_features]
    X_train, y_train = smote.fit_resample(X_selected, y_train)

    # Convert back to DataFrame
    X_train_df = pd.DataFrame(X_train, columns=X_train.columns)

    return X_train_df, y_train


# Feature Engineering
X_train, X_test, y_train, y_test = features_train_test(df)
X_train, X_test = features_transformation(X_train, X_test)

# Feature Selection
l1_features = l1_feature_selection(X_train, y_train)
print(l1_features)
tree_features_df = tree_feature_importance(X_train, y_train)
print(tree_features_df)
rfe_features = rfe_feature_selection(X_train, y_train)
print(rfe_features)

# Apply SMOTE
# after oversampling 9 785 635 each 'Is Laundering'
X_train_final, y_train_final = apply_smote(X_train, y_train)


# ---------------------------------
# Final data with selected features
# ---------------------------------

# Get the names of the selected features and Convert to DataFrame
X_train_df = pd.DataFrame(
    X_train_selected.toarray(),
    columns=selected_feature_names)
X_test_df = pd.DataFrame(
    X_test_selected.toarray(),
    columns=selected_feature_names)

# -----------------------------
# Notes
# -----------------------------

# Encode categorical variables: There is many approaches and it depends on the models that we plan to use afterwards
#   One-hot encoding -> increases dimensionality, SMOTE might create unrealistic combinations for high-cardinality features -> for distance-based or linear models (LogisticRegression, SVM, KNN)
#   Ordinal encoding -> imposes an artificial order that may mislead models that assume distance is meaningful -> for tree-based models because they don’t assume numeric distance meaning
#   Target/Mean Encoding -> replaces categories with statistics from the target

# Encode numerical variables
#   StandardScaler: uses mean and std. is sensible to
#   MinMaxScaler: uses min and max and scales the features in that range. Sensitive to outliers. Not robust to outliers. Preserves the original distribution shape
# RobustScaler: uses median and IQR instead of mean and std. less sensitive to outliers
