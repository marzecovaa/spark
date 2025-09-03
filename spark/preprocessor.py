import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocessor(input_csv: str):
    """
    This function reads the merged csv file, selects relevant columns,
    performs a simple preprocessing on questionnaire (one hot encoding)
    + demographics data
    The data is split in the train_test...
    TO DO: split train/test into a separate function?
    TO DO: data load func?
    """

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


    #encoding
    r_scaler = RobustScaler()
    mm_scaler = MinMaxScaler()
    encoder = OneHotEncoder(drop = 'if_binary')

    data_to_rscale = ['age_at_diagnosis', 'age', 'height', 'weight']
    data_to_mmscale = ['bmi']
    data_to_encode = X.drop(columns = ['age_at_diagnosis', 'age',
                                                'height', 'weight','bmi']).columns
    column_prep = ColumnTransformer(transformers=[
            ("robust", r_scaler, data_to_rscale),
            ("mm", mm_scaler, data_to_mmscale),
            ("enc",encoder, data_to_encode)
        ])

    preproc = column_prep.fit(X_train)

    X_train = preproc.transform(X_train)
    X_test = preproc.transform(X_test)

    return X_train, X_test, y_train, y_test
