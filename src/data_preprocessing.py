import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_info(df):
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.duplicated().sum())