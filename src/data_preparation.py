# 2. Cleaning and Formating the data -> check for data formats, distribution, outliers, then normalize, encode date

import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

class LoadAndClean:
    def load_to_pd(self, input_path):
        df = pd.read_csv(input_path)
        return df

    def basic_checks(self, df):
        print('--------Info--------')
        print(df.info())
        print('--------Duplicates--------')
        print(df.loc[df.duplicated()])
        print('--------Unique values in categorical variables--------')
        for col in df.select_dtypes(include='object'):
            print(col, df[col].nunique())

    # check for nulls -> no null values
    def check_nulls(self, df):
        nulls = df.isnull().sum()
        logging.info(f'Nulls values in: \n{nulls}')
        if nulls.empty:
            logging.info(f'No Null values in dataset')

    # As there are some text columns, is best to standardize, (for example, in case of a typo, extra spaces, or caps)
    def clean_text_columns(self, df):
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].str.strip().str.lower()
        return df


def merge_data(df1, df2, key1, key2, join):
    df = df1.merge(df2, left_on=key1, right_on=key2, how=join)
    return df


# Detect outliers in the transaction amount -> given that I am trying to detect for fraudulent spikes, suspicious account activity, unusual transaction,
# and those outliers might be a sign of thay, will not remove them. such patterns might be useful to train the models
def make_boxplot(df, col):
    plt.boxplot(df[col])
    plt.title('Box Plot')
    plt.show()


class Transform:
    def data_format(self, df):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.drop(columns=['Account Number'])
        df.drop(columns=['Account.1'])
        df.rename(columns={'Account': 'Account Number'}, inplace =True)
        return df



# Create the object
loader = LoadAndClean()

df_transactions = loader.load_to_pd('data/processed/transactions.csv')
loader.basic_checks(df_transactions)
loader.check_nulls(df_transactions)
df_transactions = loader.clean_text_columns(df_transactions)
df_transactions.drop_duplicates(df_transactions) # Remove duplicated as it can skew the models

df_accounts = loader.load_to_pd('data/processed/accounts.csv')
loader.basic_checks(df_accounts)
loader.check_nulls(df_accounts)
df_accounts = loader.clean_text_columns(df_accounts)

# Create merged dataset
df_merged = merge_data(df_transactions, df_accounts, 'Account', 'Account Number', 'left')

# Create the object
transformer = Transform()
df_merged = transformer.data_format(df_merged)

make_boxplot(df_merged, 'Amount Paid')

# Data validation and sanity checks for obvious errors - negative amount transaction, transaction with dates in the future
assert df_merged['Amount Paid'].min() >= 0, 'Negative transaction amounts found!'
assert (df_merged['Timestamp'] <= pd.Timestamp.today()).any(), 'Found future transactions'


df_merged.info()


#   Column              Dtype         
# ---  ------              -----         
#  0   Timestamp           datetime64[ns]
#  1   From Bank           int64         
#  2   Account Number      object        
#  3   To Bank             int64         
#  4   Amount Received     float64       
#  5   Receiving Currency  object        
#  6   Amount Paid         float64       
#  7   Payment Currency    object        
#  8   Payment Format      object        
#  9   Is Laundering       int64         
#  10  Bank Name           object        
#  11  Bank ID             int64         
#  12  Entity ID           object        
#  13  Entity Name         object 
