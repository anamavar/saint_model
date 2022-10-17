from urllib.parse import _NetlocResultMixinStr
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from typing import Dict
import config

class data():
    def __init__(self, config: config.SaintConfig) -> None:
        self.config = config
        self.data = None
        self.train_data = None
        self.test_data = None
        self.validation_data = None

    def load_csv(self):
        self.data  = pd.read_csv(self.config.project.data_address)
    
    def split_data(self):
        train_data, self.test_data = train_test_split(self.data, test_size = self.config.project.test_size)
        train_data, validation_data = train_test_split(train_data, test_size = self.config.project.validation_size)
        train_data['split'] = 'train'
        validation_data['split'] = 'validation'

        self.validation_data = validation_data
        self.train_data = pd.concat([train_data, validation_data])
