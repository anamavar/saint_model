from urllib.parse import _NetlocResultMixinStr
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from typing import Dict, List, Tuple
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
import config
from torch import Tensor
from einops import rearrange
import torch

class data():
    def __init__(self, config: config.SaintConfig) -> None:
        self.config = config
        self.data = None
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        self.censored_train_data = None
        self.censored_validation_data = None

    def load_csv(self):
        self.data  = pd.read_csv(self.config.project.data_address)
    
    def split_data(self):
        train_data, self.test_data = train_test_split(self.data, test_size = self.config.project.test_size)
        train_data, validation_data = train_test_split(train_data, test_size = self.config.project.validation_size)
        train_data['split'] = 'train'
        validation_data['split'] = 'validation'

        self.validation_data = validation_data
        self.train_data = pd.concat([train_data, validation_data])

    def split_censored_data(self):


        self.censored_train_data = self.train_data[self.train_data[self.config.project.censored_column] == 0]
        # self.train_data = self.train_data.drop(self.censored_train_data.index)

        # selfcensored_validation_data = self.validation_data[self.validation_data[self.config.project.censored_column] == 0]
        # self.validation_data = validation_data.drop(censored_validation_data.index)

        #Drop the censored column from all sets.
        self.train_data = self.train_data.drop(self.config.project.censored_column, axis = 'columns')
        self.test_data = self.test_data.drop(self.config.project.censored_column, axis = 'columns')
        # self.validation_data.drop(self.config.project.censored_column, axis = 'columns', inplace = True)

        self.censored_train_data = self.censored_train_data.drop(self.config.project.censored_column, axis = 'columns')
        # self.censored_validation_data = censored_validation_data.drop(self.config.project.censored_column, axis = 'columns')

class test_dataset(torch.utils.data.Dataset):

    def __init__(self, df:pd.DataFrame, cat_cols: List[str], con_cols: List[str], target = "test"):
        self.x = df.values
        self.y = torch.zeros(df.shape[0], dtype=torch.float32)
        self.X_categorical = self._define_tensor_features(df, cat_cols, torch.int64, None)
        self.X_continuos = self._define_tensor_features(df, con_cols, torch.float32, StandardScaler())
        self.prediction_column: Tensor = rearrange(torch.zeros(df.shape[0], dtype=torch.float32), 'n -> n 1')
    
    def __len__(self):
        return self.x.shape[0]

    # def __getitem__(self, idx):
    #     return self.x[idx]

    @staticmethod
    def _define_tensor_features(df: pd.DataFrame, cols: List[str], dtype: torch.dtype,
                                transformer: TransformerMixin = None) -> Tensor:
        """It convert a ndarray fo features in a Tensor

        :param df: Contains the data to put insert in the tensor
        :param cols: list of column names used to selected the data
        :param dtype: The type of returned Tensor
        """
        if len(cols) > 0:
            df = df.loc[:, cols].values
            if transformer:
                df = transformer.fit_transform(df)
            return torch.from_numpy(df).to(dtype=dtype)
        else:
            return rearrange(torch.zeros(df.shape[0], dtype=dtype), 'n -> n 1')

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
        """It returns two tensors one for the categorical values and another for the continuous values.
        The target is concatenated as last column or of the categorical tensors or the continuous ones
        based on the type of the target

        :param idx: numeric index of the data that we want to process
        """
        return self.X_categorical[idx], torch.cat([self.X_continuos[idx], self.prediction_column[idx]]), self.y[idx]