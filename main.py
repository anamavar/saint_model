import Models
import utils
from config import SaintConfig
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from hydra.core.config_store import ConfigStore
from lit_saint import Saint, SaintDatamodule, SaintTrainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import config
import pandas as pd

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
dataObj.split_data()

if cfg.project.model == "classification":

    model = Models.binary_classification(config = cfg)

    checkpoint_callback_pre = ModelCheckpoint(
        monitor="val_loss",
        dirpath=".",
        filename="saint-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3)

    model.saint_binary_model(data= dataObj.train_data, split_column = "split", pre_callback= [checkpoint_callback_pre, early_stopping_callback], callback= [])

    model.make_predictions(test_data = dataObj.test_data, feature_importance= True)

    model.show_metrics()

    display(pd.DataFrame(model.show_feature_importance()))

elif cfg.project.model == "regression":
    model = Models.regression_model(config= cfg)
    model.saint_regression_model(data = dataObj.train_data, split_column = "split", pre_callback= [], callback= [])

    model.make_predictions(test_data= dataObj.test_data, feature_importance= True)
    model.show_metrics()
    display(pd.DataFrame(model.show_feature_importance()))

else:
    raise Exception("Invalid problem type. Please enter regression or binary.")



