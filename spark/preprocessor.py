import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from spark.model_io import save_transformer
from spark.utils import get_channel_names, channel_wise_boss, boss_transform_data


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
    # X = df.drop(columns = ['id','label'])
    X = df.drop(columns = ['label'])
    y = df['label']
    #split train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify = y)
    X_train_ids = X_train['id']
    X_test_ids = X_test['id']
    #encoding
    r_scaler = RobustScaler()
    mm_scaler = MinMaxScaler()
    encoder = OneHotEncoder(drop = 'if_binary', handle_unknown='ignore',sparse_output=False)
    data_to_rscale = ['age_at_diagnosis', 'age', 'height', 'weight']
    data_to_mmscale = ['bmi']
    data_to_encode = X.drop(columns = ['age_at_diagnosis', 'age','height',
                                       'weight','bmi','subject_id']).columns
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
    return X_train, X_test, X_train_ids, X_test_ids, y_train, y_test


    #encoding
    r_scaler = RobustScaler()
    mm_scaler = MinMaxScaler()
    encoder = OneHotEncoder(drop = 'if_binary', handle_unknown='ignore')

    data_to_rscale = ['age_at_diagnosis', 'age', 'height', 'weight']
    data_to_mmscale = ['bmi']
    data_to_encode = X.drop(columns = ['age_at_diagnosis', 'age','height',
                                       'weight','bmi','subject_id']).columns

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



def preprocessor_reduced(input_csv: str):
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
    columns_to_keep = ['id', 'age_at_diagnosis', 'age',
                       'gender', 'appearance_in_kinship',
                       '02', '03', '09', '13', '17', '20','label']
    df = df[columns_to_keep]

    #impute age at diagnosis for missing values
    mask = df['age_at_diagnosis']==0
    df.loc[mask,'age_at_diagnosis'] = df.loc[mask,'age']
    #define X, Y
    # X = df.drop(columns = ['id','label'])
    X = df.drop(columns = ['label'])
    y = df['label']
    #split train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify = y)
    X_train_ids = X_train['id']
    X_test_ids = X_test['id']
    #encoding
    r_scaler = RobustScaler()
    mm_scaler = MinMaxScaler()
    encoder = OneHotEncoder(drop = 'if_binary', handle_unknown='ignore',sparse_output=False)
    data_to_rscale = ['age_at_diagnosis', 'age']
    data_to_encode = ['gender', 'appearance_in_kinship','02', '03', '09', '13', '17', '20']
    column_prep = ColumnTransformer(transformers=[
            ("robust", r_scaler, data_to_rscale),
            ("enc",encoder, data_to_encode)
        ])
    transformer = column_prep.fit(X_train)
    # Save transformer immediately
    save_transformer(transformer)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)
    return X_train, X_test, X_train_ids, X_test_ids, y_train, y_test


def load_q_data(input_csv: str):
    """
    This function reads the merged csv file and returns the data frame and list of ids

    Usage:
    raw_prep_dir  = '../processed_data/'
    X_data, y_data, id_list = load_q_data(raw_prep_dir + 'merged_dfq.csv')"
    """
    #load csv file
    df = pd.read_csv(input_csv,index_col=0)


    #add feature BMI
    df['bmi'] = (df.weight)/((df.height/100)**2)

    #impute age at diagnosis for missing values
    mask = df['age_at_diagnosis']==0
    df.loc[mask,'age_at_diagnosis'] = df.loc[mask,'age']

    df = df.sort_values(by = 'id')
    df = df.reset_index()

    #df_q_data = df[['id','label','age_at_diagnosis','age','bmi','height','weight','gender', 'handedness','appearance_in_kinship','01', '02', '03', '04', '05', '06', '07', '08',
    #   '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21', '22', '23', '24', '25', '26', '27', '28', '29', '30']]

    X_data=df[['id','age', 'age_at_diagnosis','bmi','height','weight','gender', 'handedness','appearance_in_kinship','01', '02', '03', '04', '05', '06', '07', '08',
       '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21', '22', '23', '24', '25', '26', '27', '28', '29', '30']]

    y_data=df[['id','label']]

    id_list = df['id']

    return X_data, y_data, id_list


def train_test_split_ids (id_list, y_data, test_size = 0.2, random_state = None, stratify = True):
    """
    Returns split y and an id list to split questionnaire and time_series data
    """
    if stratify:
        y = y_data['label']
        X_train_id, X_test_id,  y_train, y_test = train_test_split(id_list, y, test_size=test_size,random_state=random_state, stratify= y)

        return X_train_id, X_test_id, y_train, y_test


class Preprocess_Q:
    """
    This function reads q data and preprocesses them for a final analysis
    """
    def __init__(self,feature_importance = False):
        mm_scaler = MinMaxScaler()
        r_scaler = RobustScaler()
        encoder = OneHotEncoder(drop = 'if_binary', handle_unknown='ignore', sparse_output=False)


        if feature_importance:
            data_to_rscale = ['age_at_diagnosis', 'age']
            data_to_encode = ['gender', 'appearance_in_kinship','02', '03', '09', '13', '17', '20']

            self.column_prep = ColumnTransformer(transformers=[("r", r_scaler, data_to_rscale), ("enc",encoder, data_to_encode)])

        else:
            data_to_rscale = ['age_at_diagnosis', 'age', 'height', 'weight']
            data_to_mmscale = ['bmi']
            data_to_encode = ['gender', 'handedness','appearance_in_kinship','01', '02', '03', '04', '05', '06', '07', '08',
                '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21', '22', '23', '24', '25', '26', '27', '28', '29', '30']

            self.column_prep = ColumnTransformer(transformers=[("r", r_scaler, data_to_rscale), ("mm", mm_scaler, data_to_mmscale), ("enc",encoder, data_to_encode)])

    def fit(self, X):
        self.column_prep.fit(X)

        return self


    def transform(self, X):

        return pd.DataFrame(self.column_prep.transform(X), columns = self.column_prep.get_feature_names_out())


def load_timeseries_data(path: str):
    """
    This function reads timeseries files and returns xarray
    path - path to the folder with the data (time seried data have to be stored in a subfolder movement/)
    """

    file_list = pd.read_csv(Path(path) / "file_list.csv")

    # Define channel names (132 total)
    channels = get_channel_names()

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


class Preprocess_Time_Boss:

    def __init__(self, strategy = 'quantile',
                 n_bins = 4, window_step = 2, window_size = 10, word_size = 6):
        self.strategy = strategy
        self.word_size = word_size
        self.window_size = window_size
        self.window_step = window_step
        self.n_bins = n_bins

    def fit(self, time_data):

        self.boss = channel_wise_boss(time_data, strategy = self.strategy, word_size = self.word_size,
                                 window_size=self.window_size, window_step = self.window_step,
                                 n_bins = self.n_bins)
        return self

    def transform(self, time_data):

        boss_data =  boss_transform_data(self.boss, time_data)

        return boss_data




def preprocess_input(df: pd.DataFrame, transformer=None):
    """
    Function takes a DataFrame and returns tranformed features.
    Assumes df is already cleaned and structured like X
    """
    if transformer is None:
        raise ValueError("Transformer must be provided for inference.")
    return transformer.transform(df)
