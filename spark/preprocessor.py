import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from spark.model_io import save_transformer


import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


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
    df = pd.read_csv(input_cs)


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






def load_questionair_data(input_csv: str):
    """
    This function reads the merged csv file"
    """
    #load csv file
    df = pd.read_csv(input_csv,index_col=0)


    #add feature BMI
    df['bmi'] = (df.weight)/((df.height/100)**2)

    #impute age at diagnosis for missing values
    mask = df['age_at_diagnosis']==0
    df.loc[mask,'age_at_diagnosis'] = df.loc[mask,'age']

    df_questionair_data = df[['id','label','age','bmi','height','weight','gender', 'handedness','appearance_in_kinship','01', '02', '03', '04', '05', '06', '07', '08',
       '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21', '22', '23', '24', '25', '26', '27', '28', '29', '30']]

    X_q=df[['id','age','bmi','height','weight','gender', 'handedness','appearance_in_kinship','01', '02', '03', '04', '05', '06', '07', '08',
       '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21', '22', '23', '24', '25', '26', '27', '28', '29', '30']]

    y_q=df[['id','label']]

    return X_q, y_q


def questionair_preprocessor(input_csv: str):

    #load csv file
    df = load_questionair_data(input_csv)

    mm_scaler = MinMaxScaler()
    encoder = OneHotEncoder(drop = 'if_binary', handle_unknown='ignore')


    data_to_mm_scale = ['age','bmi','height','weight']
    data_to_encode = ['gender', 'handedness','appearance_in_kinship','01', '02', '03', '04', '05', '06', '07', '08',
       '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21', '22', '23', '24', '25', '26', '27', '28', '29', '30']

    column_prep = ColumnTransformer(transformers=[("mm", mm_scaler, data_to_mm_scale), ("enc",encoder, data_to_encode)])
    transformer = column_prep.fit(X_train)

    return





def load_timeseries_data(path: str):
    """
    This function reads timeseries files and returns xarray
    """

    file_list = pd.read_csv(Path(path) / "file_list.csv")

    # Define channel names (132 total)
    channels = [
        f"{task}_{sensor}_{device}_{axis}"
        for task in ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2",
                     "StretchHold", "HoldWeight", "DrinkGlas", "CrossArms",
                     "TouchNose", "Entrainment1", "Entrainment2"]
        for device in ["LeftWrist", "RightWrist"]
        for sensor in ["Acceleration", "Rotation"]
        for axis in ["X", "Y", "Z"]
    ]

    data, ids = [], []

    for subject_id in file_list["id"]:
        arr = np.fromfile(
            Path(path) / f"movement/{int(subject_id):03d}_ml.bin",
            dtype=np.float32
        ).reshape((-1, 976))  # (C, T)
        data.append(arr)
        ids.append(int(subject_id))

    X = np.stack(data)  # (N, C, T)
    timesteps = np.arange(X.shape[2])

    da = xr.DataArray(
        X,
        dims=("id", "channel", "timestep"),
        coords={
            "id": ids,
            "channel": channels,
            "timestep": timesteps,
        },
        name="timeseries"
    )

    return da





def preprocess_input(df: pd.DataFrame, transformer=None):
    """
    Function takes a DataFrame and returns tranformed features.
    Assumes df is already cleaned and structured like X
    """
    if transformer is None:
        raise ValueError("Transformer must be provided for inference.")
    return transformer.transform(df)



def split_with_ids(X, y, test_size,random_state,stratify):
    X_train,X_test, _,_ =train_test_split(X, y, test_size=test_size,random_state=random_state,  stratify= stratify)
    X_train_id = (X_train["id"]).sort_values()
    X_test_id = (X_test["id"]).sort_values()

    return X_train_id, X_test_id
