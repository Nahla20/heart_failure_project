import pandas as pd

def create_features(df):

    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[20,40,60,100],
        labels=['Young','Middle','Senior']
    )

    df['CholesterolLevel'] = pd.cut(
        df['Cholesterol'],
        bins=[0,200,240,600],
        labels=['Normal','Borderline','High']
    )

    age_chol_avg = df.groupby('AgeGroup')['Cholesterol'].mean()
    df['AgeGroup_Cholesterol_Avg'] = df['AgeGroup'].map(age_chol_avg)

    cp_hd_rate = df.groupby('ChestPainType')['HeartDisease'].mean()
    df['ChestPain_HD_Rate'] = df['ChestPainType'].map(cp_hd_rate)

    sex_maxhr_avg = df.groupby('Sex')['MaxHR'].mean()
    df['Sex_MaxHR_Avg'] = df['Sex'].map(sex_maxhr_avg)

    return df