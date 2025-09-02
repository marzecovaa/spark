import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocessor(input_csv: str):

    #load csv file
    df = pd.read_csv(input_csv)


    #drop unnecessary columns
    columns_to_drop = ['study_id',
                       'condition',
                       'disease_comment',
                       'appearance_in_first_grade_kinship',
                       'effect_of_alcohol_on_tremor',
                       'questionnaire_name',
                       'resource_type',
                       'questionnaire_id']

    df = df.drop(columns= columns_to_drop)

    #add feature BMI
    df['bmi'] = (df.weight)/((df.height/100)**2)

    #impute age at diagnosis for missing values
    mask = df['age_at_diagnosis']==0
    df.loc[mask,'age_at_diagnosis'] = df.loc[mask,'age']

    #define X, Y
    X = df.drop(columns = ['id','condition_cat'])
    y = df['condition_cat']

    #split train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify = y)

    return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = preprocessor("/Users/sebastian.zilles/code/marzecovaa/spark/processed_data/merged_dfq_v2.csv")
