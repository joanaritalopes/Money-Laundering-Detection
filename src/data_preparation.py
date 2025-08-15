# 2. Cleaning and Formating the data -> check for data formats, distribution, outliers, then normalize, encode date

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

class LoadAndClean:
    def load_to_pd(self, input_path):
        df = pd.read_csv(input_path)
        return df

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


class Transform:
    def data_format(self, df):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['From Bank'] = df['From Bank'].astype(str)
        df['To Bank'] = df['To Bank'].astype(str)
        df = df.drop(columns=['Account Number','Account.1','Bank ID'])
        df = df.rename(columns={'Account': 'Account Number'})
        return df
    
    def date_format(self, df):
        for col in ['Timestamp']:
            df[col + '_year'] = df[col].dt.year
            df[col + '_month'] = df[col].dt.month
            df[col + '_day'] = df[col].dt.day
            df[col + '_weekday'] = df[col].dt.weekday
        df = df.drop(columns=['Timestamp'])
        return df


# Create the object
loader = LoadAndClean()

df_transactions = loader.load_to_pd('data/processed/transactions.csv')
loader.check_nulls(df_transactions)
df_transactions = loader.clean_text_columns(df_transactions)
df_transactions.drop_duplicates(df_transactions) # Remove duplicated as it can skew the models

df_accounts = loader.load_to_pd('data/processed/accounts.csv')
loader.check_nulls(df_accounts)
df_accounts = loader.clean_text_columns(df_accounts)

# Create merged dataset
df_merged = merge_data(df_transactions, df_accounts, 'Account', 'Account Number', 'left')

# Create the object
transformer = Transform()
df_merged = transformer.data_format(df_merged)

# Data validation and sanity checks for obvious errors - negative amount transaction, transaction with dates in the future
assert df_merged['Amount Paid'].min() >= 0, 'Negative transaction amounts found!'
assert (df_merged['Timestamp'] <= pd.Timestamp.today()).any(), 'Found future transactions'

df_final = transformer.date_format(df_merged)

# Save as csv to perform EDA
df_final.to_csv('data/processed/df_final.csv', index=False)
