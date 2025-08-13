# 5. Feature Engineering

from src.data_preparation import df_merged

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


df_merged = flag_outliers(df_merged, 'Amount Paid')
df_merged.columns

#df_merged.loc[df_merged['is_outlier'] == 1].tail() # Check the outliers rows




# Encode categorical vals

# Extract date parts

# Scaling & Normalization

# Deal with imbalanced data

#this is an Imbalanced dataset, so SMOTE generate rows for the minority class

# from imblearn.over_sampling import SMOTE

# attrs=df_merged.drop(['Is Laundering'],axis=1)
# target=df_merged['Is Laundering']

# smote = SMOTE(sampling_strategy='minority')
# attrs, target  = smote.fit_resample(attrs, target)
# target.value_counts()

# Train-Test split