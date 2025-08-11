# 2. Cleaning and Formating the data -> check for data formats, distribution, outliers, then normalize, encode date

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df_transactions = pd.read_csv('data/processed/transactions.csv')
df_accounts = pd.read_csv('data/processed/accounts.csv')
print(df_transactions)

# data types and nulls - to convert and fill null values
df_transactions.info()

#     Column              Dtype  
# --  ------              -----  
# 0   Timestamp           object 
# 1   From Bank           int64  
# 2   Account             object 
# 3   To Bank             int64  
# 4   Account.1           object 
# 5   Amount Received     float64
# 6   Receiving Currency  object 
# 7   Amount Paid         float64
# 8   Payment Currency    object 
# 9   Payment Format      object 
# 10  Is Laundering       int64  

df_accounts.info()

#     Column           Dtype 
# --  ------          ----- 
# 0   Bank Name       object
# 1   Bank ID         int64 
# 2   Account Number  object
# 3   Entity ID       object
# 4   Entity Name     object

# check for NULLS -> no null values
df_transactions.isnull().sum() 
df_accounts.isnull().sum()

# Check for number of unique values in categorical variables
for col in df_accounts.select_dtypes(include='object'):
    print(col, df_accounts[col].nunique())

# Check for duplicates -> there are 17 transcations duplicated
df_transactions[df_transactions.duplicated()]


# DATA FORMATING

# As there are some text columns, is best to standardize, (for example, in case of a typo, extra spaces, or caps)
df_accounts.tail()
df_transactions.tail()

for col in df_accounts.select_dtypes(include='object'):
    df_accounts[col] = df_accounts[col].str.strip().str.lower()

for col in df_transactions.select_dtypes(include='object'):
    df_transactions[col] = df_transactions[col].str.strip().str.lower()


df_merged = df_transactions.merge(df_accounts, left_on='Account', right_on='Account Number', how='left')
df_merged

# Check for outliers in the transaction amount -> given that I am trying to detect for fraudulent spikes, suspicious account activity, unusual transaction,
# and those outliers might be a sign of thay, will not remove them. such patterns might be useful to train the models

plt.boxplot(df_merged['Amount Paid'])
plt.title('Box Plot of Transactions')
plt.show()

# Instead of removing the outliers, we will flag them by adding a binary column using Interquartile Range Method
q1 = df_merged['Amount Paid'].quantile(0.25)
q3 = df_merged['Amount Paid'].quantile(0.75)
iqr = q3 - q1
# Determine outlier boundaries
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df_merged['is_outlier'] = (
    (df_merged['Amount Paid'] < lower_bound) or
    (df_merged['Amount Paid'] > upper_bound)
).astype(int)

# Check the outliers rows
df_merged.loc[df_merged['is_outlier'] == 1].tail()