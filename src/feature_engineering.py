# 5. Feature Engineering

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from src.data_preparation import df_final

# Instead of removing the outliers, we will flag them by adding a binary column using Interquartile Range Method
def flag_outliers(df, col, multiplier=1.5):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    # Determine outlier boundaries
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df['Is Outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
    return df

df_final = flag_outliers(df_final, 'Amount Paid')
df_final.loc[df_final['Is Outlier'] == 1].tail() # Check the outliers rows

# 'Timestamp', 'From Bank', 'Account Number', 'To Bank',
    # 'Amount Received', 'Receiving Currency', 'Amount Paid',
    # 'Payment Currency', 'Payment Format', 'Is Laundering', 'Bank Name',
    # 'Entity ID', 'Entity Name', 'Is Outlier'

df = df_final.copy()


# Split features from target variables
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col != 'Is Laundering']
TARGET_COL = 'Is Laundering'

# Split into train/test
X = df.drop(columns=[TARGET_COL], axis=1)
y = df[TARGET_COL]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------------------------------------------------------
# Encode categorical variables: There is many approaches and it depends on the models that we plan to use afterwards
#   One-hot encoding -> increases dimensionality, SMOTE might create unrealistic combinations for high-cardinality features -> for distance-based or linear models (LogisticRegression, SVM, KNN)
#   Ordinal encoding -> imposes an artificial order that may mislead models that assume distance is meaningful -> for tree-based models because they don’t assume numeric distance meaning
#   Target/Mean Encoding -> replaces categories with statistics from the target
# ------------------------------------------------------------------------------------------------------------------
# As data is already splitted into train/test because while fitting an encoder on the entire dataset before, 
#information from the test set is “seen” during encoding which might lead to data leakage.
# ------------------------------------------------------------------------------------------------------------------

# Numerical data is right-skwed and it has outliers which decided not to remove them
# StandardScaler uses mean and std. is sensible to 
# MinMaxScaler uses min and max and scales the features in that range. Sensitive to outliers. Not robust to outliers. Preserves the original distribution shape
# RobustScaler uses median and IQR instead of mean and std. less sensitive to outliers

# Encode categorical variables + Scale numerical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', RobustScaler(), numerical_cols)
    ]
)

smote = SMOTE(sampling_strategy='minority', random_state=42)

# Apply preprocessing + SMOTE on train set
X_train_encoded = preprocessor.fit_transform(X_train)

X_train_transformed, y_train_res = smote.fit_resample(X_train_encoded, y_train) # type: ignore

y_train_res.value_counts() # 9 785 635 each 'Is Laundering'

X_test_transformed = preprocessor.transform(X_test) # Transform test set - no SMOTE on test


# -------------------
# Try several models
# -------------------
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}

for name, model in models.items():
    model.fit(X_train_transformed, y_train_res)
    score = model.score(X_test_transformed, y_test)
    print(f"{name} test accuracy: {score:.4f}")


# Approach for the decided model

# pipeline = Pipeline(
#     steps=[
#         ('preprocessor', preprocessor),
#         ('smote', SMOTE(random_state=42)), # Deal with imbalanced data - generate rows for the minority class
#         ('model', RandomForestClassifier(random_state=42))
#     ]
# )
