import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import classification_report, explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
# import wget
from lit_saint import Saint, SaintConfig, SaintDatamodule, SaintTrainer
from pytorch_lightning import Trainer, seed_everything
from typing import Dict
import config

class binary_classification():
    def __init__(self, config: config.SaintConfig) -> None:
       self.trainer = None
       self.model = None
       self.data_module = None
       self.predictions = None
       self.config = config

    def saint_binary_model(self, data: pd.DataFrame, split_column: str, pre_callback: list = [], callback: list = []):
        """This function defines and trains a Saint model for binary classification

        Args:
            data (Pandas.DataFrame): A DataFrame includes training and validation data.
            split_column (string): Name of the column that shows training and validation data in the DataFrame
        """        
        self.data_module = SaintDatamodule(df = data, target = data.columns[self.config.project.target_column], split_column = split_column)
        self.model = Saint(categories = self.data_module.categorical_dims, continuous = self.data_module.numerical_columns, dim_target = self.data_module.dim_target, config = self.config)
        pretrainer = Trainer(max_epochs = self.config.pretrain.epochs, callbacks = pre_callback)
        trainer = Trainer(max_epochs = self.config.train.epochs, callbacks = callback)
        self.trainer = SaintTrainer(pretrainer = pretrainer, trainer = trainer, pretrain_loader_params = {"batch_size": self.config.pretrain.batch_size}, train_loader_params = {"batch_size": self.config.train.batch_size})
        self.trainer.fit(model = self.model, datamodule= self.data_module, enable_pretraining= True)
    
    def make_predictions(self, test_data: pd.DataFrame, feature_importance: bool = False):
        """Use the trained model to do the prediction over test data

        Args:
            test_data (Pandas.DataFrame): A DataFrame consists of test samples. 
        """        

        self.predictions = self.trainer.predict(model = self.model, datamodule = self.data_module, df = test_data, feature_importance = feature_importance)

    def show_metrics(self):
        """This function shows the metrics for the classification, including accuracy, precision, recall, and f-measure.
        """        
        print(classification_report(self.data_module.predict_set[self.data_module.target], np.argmax(self.predictions['prediction'], axis=1)))
    
    def show_feature_importance(self):

        imp = []

        pred_cls = np.argmax(self.predictions['prediction'], axis=1)

        for label in np.unique(pred_cls):
            tmp = self.predictions['importance'][pred_cls == label]
            imp.append(tmp.mean(axis = 0).sort_values())
        return imp 


class regression_model():

    def __init__(self, config: config.SaintConfig) -> None:
        self.trainer = None
        self.model = None
        self.data_module = None
        self.predictions = None
        self.config = config

    def saint_regression_model(self, data: pd.DataFrame, split_column: str, pre_callback: list = [], callback: list = []):
        """This function defines and trains a Saint model for regression problems

        Args:
            data (Pandas.DataFrame): A DataFrame includes training and validation data.
            split_column (string): Name of the column that shows training and validation data in the DataFrame
            config (obj): The object of config includes all the config(setting) file information.
        """        
        self.data_module = SaintDatamodule(df = data, target = data.columns[self.config.project.target_column], split_column= split_column)
        self.model = Saint(categories= self.data_module.categorical_dims, continuous= self.data_module.numerical_columns, config = self.config, dim_target= self.data_module.dim_target)
        pretrainer = Trainer(max_epochs= self.config.pretrain.epochs)
        trainer = Trainer(max_epochs= self.config.train.epochs)
        self.trainer = SaintTrainer(pretrainer= pretrainer, trainer= trainer, pretrain_loader_params = {"batch_size": self.config.pretrain.batch_size}, train_loader_params = {"batch_size": self.config.train.batch_size})
        self.trainer.fit(model = self.model, datamodule= self.data_module, enable_pretraining= True)

    def make_predictions(self, test_data: pd.DataFrame, feature_importance: bool = False):
        """Use the trained model to do the prediction over test data

        Args:
            test_data (Pandas.DataFrame): A DataFrame consists of test samples. 
        """  
        self.predictions = self.trainer.predict(model = self.model, datamodule= self.data_module, df = test_data, feature_importance = feature_importance)

    def show_metrics(self):
        """This function shows the variance, MAE, and MSE for the predction
        """        
        expl_variance = explained_variance_score(self.data_module[self.data_module.target], self.predictions['prediction'])
        mae = mean_absolute_error(self.data_module[self.data_module.target], self.predictions['prediction'])
        mse = mean_squared_error(self.data_module[self.data_module.target], self.predictions['prediction'])
        print(f"Explained Variance: {expl_variance} MAE: {mae} MSE: {mse}")

    def show_feature_importance(self):

        imp = []
        pred_cls = np.argmax(self.predictions['prediction'], axis=1)

        for label in np.unique(pred_cls):
            tmp = self.predictions['importance'][pred_cls == label]
            imp.append(tmp.mean(axis = 0).sort_values())
        return imp 



        
        