# 4. EDA
# 5. Feature Engineering


# Instead of removing the outliers, we will flag them by adding a binary column using Interquartile Range Method
def flag_outliers(df, col, multiplier=1.5):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    # Determine outlier boundaries
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df['is_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
    return df


#flag_outliers(df_merged, 'Amount Paid')
#df_merged.loc[df_merged['is_outlier'] == 1].tail() # Check the outliers rows




# Encode categorical vals

# Extract date parts

# Scaling & Normalization

# Train-Test split