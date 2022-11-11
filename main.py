import Models
import utils
import torch
from config import SaintConfig
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from hydra.core.config_store import ConfigStore
from lit_saint import Saint, SaintDatamodule, SaintTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import config
import pandas as pd
import os
import datetime


def save_model(cfg, file_name):
    dt = datetime.datetime.now()
    dt = dt.strftime("%d_%m_%Y_%H_%M_%S")
    file_name = file_name + dt
    save_path = os.path.join(cfg.project.save_dir, file_name)
    torch.save(model, save_path)

cs = ConfigStore.instance()
cs.store(name="base_config", node=SaintConfig)
with initialize(config_path="."):
    cfg = compose(config_name="config")

cfg.network.transformer.attention_type = config.AttentionTypeEnum(cfg.network.transformer.attention_type)
cfg.pretrain.task.contrastive.contrastive_type = config.ContrastiveEnum(cfg.pretrain.task.contrastive.contrastive_type)
cfg.pretrain.task.contrastive.projhead_style = config.ProjectionHeadStyleEnum(cfg.pretrain.task.contrastive.projhead_style)
cfg.pretrain.task.denoising.denoising_type = config.DenoisingEnum(cfg.pretrain.task.denoising.denoising_type)




dataObj = utils.data(config= cfg)
# Load and split the dataset into train, test, and validation sets.
dataObj.load_csv()




if cfg.project.mode == "train":
    
    if type(cfg.project.target_column) is int:
        cfg.project.target_column = dataObj.data.columns[cfg.project.target_column]

    dataObj.split_data()

    if cfg.project.model == "classification":

        model = Models.saint_classification(config = cfg)

        checkpoint_callback_pre = ModelCheckpoint(
            monitor="val_loss",
            dirpath=".",
            filename="saint-{epoch:02d}-{val_loss:.2f}",
            mode="min",
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3)

        model.saint_classification_model(data= dataObj.train_data, split_column = "split", pre_callback= [checkpoint_callback_pre, early_stopping_callback], callback= [])

        save_model(cfg, "classification")

        model.make_predictions(test_data = dataObj.test_data, feature_importance= True)

        model.show_metrics()

        display(pd.DataFrame(model.show_feature_importance()))

    elif cfg.project.model == "regression":
        model = Models.regression_model(config= cfg)
        model.saint_regression_model(data = dataObj.train_data, split_column = "split", pre_callback= [], callback= [])

        save_model(cfg, "regression")

        model.make_predictions(test_data= dataObj.test_data, feature_importance= False)
        model.show_metrics()
        # display(pd.DataFrame(model.show_feature_importance()))

    elif cfg.project.model == "censored":
        dataObj.split_censored_data()
        model = Models.Censored_Model(config = cfg)
        model.saint_censored_model(data = dataObj.train_data, censored_data = dataObj.censored_train_data, split_column = 'split')
        save_model(cfg, "censored")

        model.make_predictions(test_data= dataObj.test_data, feature_importance= False)
        model.show_metrics()

    else:
        raise Exception("Invalid problem type. Please enter regression or binary.")
elif cfg.project.mode == "test":

    # make fake columns
    dataObj.data['label'] = 0
    dataObj.data['split'] = 'train'
    
    if cfg.project.model == "classification":
        # load the model and create a trainer
        clf = torch.load(cfg.project.saved_model)

        # build a data module based on the data
        clf.data_module = SaintDatamodule(df = dataObj.data, target = 'label', split_column = 'split')

        clf.make_predictions(test_data= dataObj.data, feature_importance= True)
        print(clf.predictions['prediction'])

        display(pd.DataFrame(clf.show_feature_importance()))
    
    elif cfg.project.model == "regression":
        #Load the model
        clf = torch.load(cfg.project.saved_model)

        # build a data module based on the data
        clf.data_module = SaintDatamodule(df = dataObj.data, target = 'label', split_column = 'split')

        clf.make_predictions(test_data= dataObj.data, feature_importance= False)
        print(clf.predictions['prediction'])
        
        display(clf.show_feature_importance())

    elif cfg.project.model == "censored":
        clf = torch.load(cfg.project.saved_model)

        # build a data module based on the data
        clf.data_module = SaintDatamodule(df = dataObj.data, target = 'label', split_column = 'split')

        clf.make_predictions(test_data= dataObj.data, feature_importance= False)
        print(clf.predictions['prediction'])
        
        display(clf.show_feature_importance())
