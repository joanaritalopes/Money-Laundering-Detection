# 2. Cleaning and Formating the data -> check for data formats,
# distribution, outliers, then normalize, encode date

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


class LoadAndClean:
    '''Loading CSV data and performing basic cleaning operations.'''

    def load_to_pd(self, input_path: str) -> pd.DataFrame:
        '''
        Load a CSV file into a pandas DataFrame.

        Parameters
        ----------
        input_path : str
            The path to the CSV file.

        Returns
        -------
        pd.DataFrame with loaded data.
        '''

        df = pd.read_csv(input_path)

        return df

    def check_nulls(self, df: pd.DataFrame) -> None:
        '''
        Check for null values in the DataFrame and log the results.

        Parameters
        ----------
        pd.DataFrame to check for nulls.

        Returns
        -------
        None
        '''

        nulls = df.isnull().sum()
        logging.info('Null values in:\n%s', nulls)
        if nulls.empty:
            logging.info('No null values in dataset')

    def clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Standardize text columns by stripping whitespace and converting to lowercase as there are some
        text columns (for example, in case of a typo, extra spaces, or caps).

        Parameters
        ----------
        pd.DataFrame with text columns to clean.

        Returns
        -------
        pd.DataFrame
            DataFrame with cleaned text columns.
        '''

        for col in df.select_dtypes(include='object'):
            df[col] = df[col].str.strip().str.lower()

        return df


def merge_data(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        key1: str,
        key2: str,
        join) -> pd.DataFrame:
    '''
    Merge two DataFrames on specific keys and join type.

    Parameters
    ----------
    df1 : pd.DataFrame
        Left DataFrame.
    df2 : pd.DataFrame
        Right DataFrame.
    key1 : str
        column in df1 to join on.
    key2 : str
        column in df2 to join on.
    join : str
        Type of join: 'inner', 'left', 'right', or 'outer'.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame.
    '''

    merged_df = df1.merge(df2, left_on=key1, right_on=key2, how=join)

    return merged_df


class Transform:
    '''Transform and format DataFrame columns.'''

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Convert column types and drop/rename unnecessary columns.

        Parameters
        ----------
        pd.DataFrame to format.

        Returns
        -------
        pd.DataFrame: formatted df with appropriate column types.
        '''

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['From Bank'] = df['From Bank'].astype(str)
        df['To Bank'] = df['To Bank'].astype(str)
        df = df.drop(columns=['Account Number', 'Account.1', 'Bank ID'])
        df = df.rename(columns={'Account': 'Account Number'})

        return df

    def extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Extract day and weekday from the 'Timestamp' column and drop the original column.

        Parameters
        ----------
        pd.DataFrame containing a 'Timestamp' column.

        Returns
        -------
        pd.DataFrame with added 'Timestamp_day' and 'Timestamp_weekday' columns.
        '''

        for col in ['Timestamp']:
            df[col + '_day'] = df[col].dt.day
            df[col + '_weekday'] = df[col].dt.weekday
        df = df.drop(columns=['Timestamp'])

        return df


# Create the object
loader = LoadAndClean()

df_transactions = loader.load_to_pd('data/processed/transactions.csv')
loader.check_nulls(df_transactions)
df_transactions = loader.clean_text_columns(df_transactions)
# Remove duplicated as it can skew the models
df_transactions.drop_duplicates(df_transactions)

df_accounts = loader.load_to_pd('data/processed/accounts.csv')
loader.check_nulls(df_accounts)
df_accounts = loader.clean_text_columns(df_accounts)

# Create merged dataset
df_merged = merge_data(
    df_transactions,
    df_accounts,
    'Account',
    'Account Number',
    'left')

# Create the object
transformer = Transform()
df_merged = transformer.format_data(df_merged)

# Data validation and sanity checks for obvious errors - negative amount
# transaction, transaction with dates in the future
assert df_merged['Amount Paid'].min() >= 0, 'Negative transaction amounts found!'
assert (df_merged['Timestamp'] <= pd.Timestamp.today()).any(), 'Found future transactions'

df_final = transformer.extract_date_features(df_merged)

# Save as csv to perform EDA
df_final.to_csv('data/processed/df_final.csv', index=False)
