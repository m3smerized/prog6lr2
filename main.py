import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv('train.csv', header=0, names=[
    'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex',
    'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
])

def task_1_sex_count(df):
    male_count = (df['Sex'] == 'male').sum()
    female_count = (df['Sex'] == 'female').sum()
    print(f"1) {male_count} {female_count}")

def task_2_embarked_count(df):
    ports = df['Embarked'].fillna('Unknown')
    s = (ports == 'S').sum()
    c = (ports == 'C').sum()
    q = (ports == 'Q').sum()
    print(f"2) {s} {c} {q}")

def task_3_death_rate(df):
    total = df.shape[0]
    deaths = (df['Survived'] == 0).sum()
    rate = round((deaths / total) * 100, 3)
    print(f"3){deaths} {rate}")

def task_4_class_share(df):
    total = df.shape[0]
    class_counts = df['Pclass'].value_counts().sort_index()
    percents = [round((class_counts[i] / total) * 100, 3) for i in [1, 2, 3]]
    print(f"4) {percents[0]} {percents[1]} {percents[2]}")

def task_5_corr_sibsp_parch(df):
    print(f"5) {df[['SibSp', 'Parch']].corr().iloc[0,1]:.5f}")

def task_6_corr_survived(df):
    age_corr = df[['Survived', 'Age']].dropna().corr().iloc[0,1]
    sex_mapped = df['Sex'].map({'male': 0, 'female': 1})
    sex_corr = df['Survived'].corr(sex_mapped)
    class_corr = df['Survived'].corr(df['Pclass'])
    print(f"6) {age_corr:.5f}")
    print(f"{sex_corr:.5f}")
    print(f"{class_corr:.5f}")

def task_7_age_stats(df):
    age = df['Age'].dropna()
    print(f"7) {round(age.mean(),3)} {round(age.median(),3)} {age.min()} {age.max()}")

def task_8_fare_stats(df):
    fare = df['Fare'].dropna()
    print(f"8) {round(fare.mean(),3)} {round(fare.median(),3)} {fare.min()} {fare.max()}")

def extract_names(df):
    parts = df['Name'].str.extract(r'(?P<Last>[^,]+),\s(?P<Title>[^.]+)\.\s(?P<First>.+)')
    return pd.concat([parts, df['Sex'], df['Age']], axis=1)

def task_9_popular_male_name(df):
    names = extract_names(df)
    males = names[names['Sex'] == 'male']['First']
    males_clean = males.str.extract(r'(\w+)')[0]
    top = males_clean.value_counts().idxmax()
    print(f"9) {top}")

def task_10_popular_names_older_15(df):
    coltitle = (df['Name']
                .apply(lambda s: pd.Series(
                    {'Title': s.split(',')[1].split('.')[0].strip(),
                    'LastName':s.split(',')[0].strip(),
                    'FirstName':s.split(',')[1].split('.')[1].strip()})))
    joined = coltitle.join(df[['Age', 'Sex']])
    print('10) Самые популярные мужские и женские имена старше 15 лет')
    print(joined[joined['Age'] > 15].value_counts(subset=['FirstName', 'Sex']))



task_1_sex_count(df)
task_2_embarked_count(df)
task_3_death_rate(df)
task_4_class_share(df)
task_5_corr_sibsp_parch(df)
task_6_corr_survived(df)
task_7_age_stats(df)
task_8_fare_stats(df)
task_9_popular_male_name(df)
task_10_popular_names_older_15(df)
