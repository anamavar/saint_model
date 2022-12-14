from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum


class ContrastiveEnum(Enum):
    simsiam = 'simsiam'
    standard = 'standard'
    disabled = 'disabled'


class DenoisingEnum(Enum):
    standard = 'standard'
    disabled = 'disabled'


class ProjectionHeadStyleEnum(Enum):
    same = 'same'
    different = 'different'


class AttentionTypeEnum(Enum):
    col = "col"
    row = "row"
    colrow = "colrow"


@dataclass
class OptimizerConfig:
    """Define the parameters for CutMix augmentation"""
    learning_rate: float = 0.0001  #: value used to specify the learning rate
    other_params: Optional[Dict] = field(default_factory=dict)


@dataclass
class CutMixConfig:
    """Define the parameters for CutMix augmentation"""
    lam: float = 0.1  #: probability original values will be updated


@dataclass
class MixUpConfig:
    """Define the parameters for CutMix augmentation"""
    lam: float = 0.1  #: weight used for the linear combination


@dataclass
class ContrastiveConfig:
    """Define the parameters for Contrastive pretraining task"""
    dropout: float = .0  #: probability dropout in projection head
    contrastive_type: ContrastiveEnum = ContrastiveEnum.simsiam  # type of contrastive task to apply
    projhead_style: ProjectionHeadStyleEnum = ProjectionHeadStyleEnum.different  #: it is used to project embeddings
    nce_temp: float = 0.5  #: temperature used for the logits in case of standard constrastive type
    weight: float = 0.1  #: weight of the loss for this pretraining task


@dataclass
class DenoisingConfig:
    """Define the parameters for Denoising pretraining task"""
    denoising_type: DenoisingEnum = DenoisingEnum.standard  # type of denoising task to apply
    weight_cross_entropy: float = 0.5  #: weight reconstruction loss for categorical features
    weight_mse: float = 0.5  #: weight reconstruction loss for continuous features
    scale_dim_internal_sepmlp: float = 5  # scale factor of the input dimension for the first linear layer
    dropout: float = .0  #: probability dropout in SepMLP


@dataclass
class AugmentationConfig:
    """Define the parameters used for the augmentations"""
    cutmix: Optional[CutMixConfig] = CutMixConfig()
    mixup: Optional[MixUpConfig] = MixUpConfig()


@dataclass
class PreTrainTaskConfig:
    """Define the parameters used for pretraining tasks"""
    contrastive: Optional[ContrastiveConfig] = ContrastiveConfig()
    denoising: Optional[DenoisingConfig] = DenoisingConfig()


@dataclass
class PreTrainConfig:
    """Define parameters for the steps used during the pretraining"""
    aug: AugmentationConfig = AugmentationConfig()
    task: PreTrainTaskConfig = PreTrainTaskConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    epochs: int = 2  #: number of epochs of training phase
    batch_size: int = 256  #: dimension of batches using by dataloaders


@dataclass
class TrainConfig:
    """Define parameters for the steps used during the training"""
    internal_dimension_output_layer: int = 20  #: internal dimension of the MLP that compute the output
    mlpfory_dropout: float = .0  #: probability dropout in the the MLP used for prediction
    epochs: int = 10  #: number of epochs of training phase
    optimizer: OptimizerConfig = OptimizerConfig()
    batch_size: int = 32  #: dimension of batches using by dataloaders


@dataclass
class TransformerConfig:
    depth: int = 3  #: number of attention blocks used in the transformer
    heads: int = 1  #: number of heads used in the transformer
    dropout: float = 0.1  #: probability dropout in the transformer
    attention_type: AttentionTypeEnum = AttentionTypeEnum.col  #: type of attention
    dim_head: int = 64
    scale_dim_internal_col: float = 4  # scale factor of the input dimension in case of attention_type col
    scale_dim_internal_row: float = 4  # scale factor of the input dimension in case of attention_type row


@dataclass
class NetworkConfig:
    """Define the neural network parameters"""
    transformer: TransformerConfig = TransformerConfig()
    num_workers: int = 0  #: number of cores to use
    embedding_size: int = 10  #: dimension of computed embeddings
    internal_dimension_embed_continuous: int = 100  #: internal dimension of the mlp used to project continuous columns
    dropout_embed_continuous: float = .0  #: dropout used to compute embedding continuous features


@dataclass
class ProjectConfig:
    """Define the project configuration"""
    data_address: str = "/data/adult.csv" # address of the dataset
    target_column: int = 14 # target column in the dataset
    test_size: float = 0.2 # portion of test data used in spliting the data
    validation_size: float = 0.1 # portion of the validation data used in spliting the dat6a
    mode: str = "train"
    save_dir: str = "/output"
    model: str = "classification"
    censored_column= ""
    saved_model = ""

@dataclass
class SaintConfig:
    """Define all the parameters used in SAINT"""
    network: NetworkConfig = NetworkConfig()
    pretrain: PreTrainConfig = PreTrainConfig()
    train: TrainConfig = TrainConfig()
    project: ProjectConfig = ProjectConfig()
