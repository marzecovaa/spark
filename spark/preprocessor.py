import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from spark.model_io import save_transformer


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
    columns_to_drop = ['study_id_x',
                       'study_id_y',
                       'condition',
                       'disease_comment',
                       'appearance_in_first_grade_kinship',
                       'effect_of_alcohol_on_tremor',
                       'questionnaire_name',
                       'resource_type',
                       'questionnaire_id','Unnamed: 0']

    df = df.drop(columns= columns_to_drop)

    #add feature BMI
    df['bmi'] = (df.weight)/((df.height/100)**2)

    #impute age at diagnosis for missing values
    mask = df['age_at_diagnosis']==0
    df.loc[mask,'age_at_diagnosis'] = df.loc[mask,'age']

    #define X, Y
    X = df.drop(columns = ['id','label'])
    y = df['label']

    #split train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify = y)


    #encoding
    r_scaler = RobustScaler()
    mm_scaler = MinMaxScaler()
    encoder = OneHotEncoder(drop = 'if_binary', handle_unknown='ignore')

    data_to_rscale = ['age_at_diagnosis', 'age', 'height', 'weight']
    data_to_mmscale = ['bmi']
    data_to_encode = X.drop(columns = ['age_at_diagnosis', 'age',
                                                'height', 'weight','bmi']).columns
    column_prep = ColumnTransformer(transformers=[
            ("robust", r_scaler, data_to_rscale),
            ("mm", mm_scaler, data_to_mmscale),
            ("enc",encoder, data_to_encode)
        ])

    transformer = column_prep.fit(X_train)

    # Save transformer immediately
    save_transformer(transformer)

    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)

    return X_train, X_test, y_train, y_test

def preprocess_input(df: pd.DataFrame, transformer=None):
    """
    Function takes a DataFrame and returns tranformed features.
    Assumes df is already cleaned and structured like X
    """
    if transformer is None:
        raise ValueError("Transformer must be provided for inference.")
    return transformer.transform(df)
