from typing import Tuple, Union, List

import torch
from torch.nn.modules import Linear
from torch.optim.adam import Adam
import torch.utils
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import lightning as pl

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MultilabelAveragePrecision,
    MulticlassConfusionMatrix,
)

from utils.zero_shot_func import (
    _create_zero_shot_classifier, 
    _evaluate_zero_shot
)

class ZeroShotCallback(pl.Callback):
    def __init__(
            self,
            dataset_name: str,
            dataloader: DataLoader,
            classnames: List[str],
            templates: List = None,
            tokenizer=None,
            text_forward=None,
            modality_forward=None,
            batch_size: int = 64,
            device: Union[str, torch.device] = "cuda",
            top_k: Tuple[int, ...] = (1, 2, 5, 10),
            average: str = "micro",
            dtype: torch.dtype = torch.float32,
            confusion_matrix: bool = False,
            multi_label: bool = False,
            verbose: bool = False
        ) -> None:
        super().__init__()

        self.dataloader = dataloader
        self.dataset_name = dataset_name
        self.classnames = classnames
        self.templates = templates
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.top_k = top_k
        self.average = average
        self.dtype = dtype
        self.confusion_matrix = confusion_matrix
        self.multi_label = multi_label
        self.verbose = verbose
        self.text_forward = text_forward
        self.modality_forward = modality_forward

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.classifier = _create_zero_shot_classifier(
            forward_func=self.text_forward,
            classnames=self.classnames,
            templates=self.templates,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
        )

        result = _evaluate_zero_shot(
            forward_func=self.modality_forward,
            classifier=self.classifier,
            dataloader=self.dataloader,
            confusion_matrix=self.confusion_matrix,
            top_k=self.top_k,
            average=self.average,
            multi_label=self.multi_label,
            device=self.device,
            dtype=self.dtype,
            verbose=self.verbose
        )

        for k, v in result.items():
            if k == "ConfusionMatrix":
                trainer.logger.log_image(
                    key=f"{self.dataset_name}-confusionmatrix", images=[v], caption=["ConfMatrix"]
                )
            else:
                trainer.logger.log(f"{self.dataset_name}-accuracy", v, sync_dist=False)

