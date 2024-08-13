from typing import Tuple, Union

import torch
from torch.nn.modules import Linear
from torch.optim.adam import Adam
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from tqdm import tqdm
import lightning as pl

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MultilabelAveragePrecision,
    MulticlassConfusionMatrix,
)

@phillip
