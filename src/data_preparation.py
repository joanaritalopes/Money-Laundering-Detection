# 2. Cleaning and Formating the data -> check for data formats, distribution, outliers, then normalize, encode date

import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def load_to_pd():
    df_transactions = pd.read_csv('data/processed/transactions.csv')
    df_accounts = pd.read_csv('data/processed/accounts.csv')
    return df_transactions, df_accounts

df_transactions, df_accounts = load_to_pd()
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

# check for nulls -> no null values
def check_nulls(df):
    nulls = df.isnull().sum()
    logging.info(f'Nulls values in: \n{nulls}')
    if nulls.empty:
        logging.info(f'No Null values in dataset')

check_nulls(df_transactions)

# Check for number of unique values in categorical variables
for col in df_accounts.select_dtypes(include='object'):
    print(col, df_accounts[col].nunique())

# Check for duplicates -> remove the 17 transcations duplicated as it can skew the models
df_transactions.loc[df_transactions.duplicated()]
df_transactions = df_transactions.drop_duplicates()

# DATA FORMATING

# As there are some text columns, is best to standardize, (for example, in case of a typo, extra spaces, or caps)
def clean_text_columns(df):
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].str.strip().str.lower()
    return df

clean_text_columns(df_transactions)
clean_text_columns(df_accounts)

df_merged = df_transactions.merge(df_accounts, left_on='Account', right_on='Account Number', how='left')
df_merged

# Detect outliers in the transaction amount -> given that I am trying to detect for fraudulent spikes, suspicious account activity, unusual transaction,
# and those outliers might be a sign of thay, will not remove them. such patterns might be useful to train the models

plt.boxplot(df_merged['Amount Paid'])
plt.title('Box Plot of Transactions')
plt.show()

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

flag_outliers(df_merged, 'Amount Paid')
df_merged.loc[df_merged['is_outlier'] == 1].tail() # Check the outliers rows

# Fix data types - dates to timestamp
----

# Data validation and sanity checks for obvious errors - negative amount transaction, transaction with dates in the future
assert df_transactions['Amount Paid'].min() >= 0, 'Negative transaction amounts found!'
assert df_transactions['Timestamp'].to_timestamp <= pd.Timestamp.today(), 'Future transactions'


