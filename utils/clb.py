import gc
from typing import List, Tuple, Union, Optional, Callable

import lightning.pytorch as pl
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.evaluation.zero_shot.zero_shot_data import get_classes, get_templates
from src.evaluation.zero_shot.zero_shot_func import _create_zero_shot_classifier, _evaluate_zero_shot


class TestCallback(pl.Callback):
    def __init__(self,
                 perform_on_validation: bool = True,
                 perform_on_test: bool = False,
                 delay_epochs: int = 0,
                 ):
        self.perform_on_validation = perform_on_validation
        self.perform_on_test = perform_on_test
        self.delay_epochs = delay_epochs

        self.dataloader = None

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.perform_on_validation and trainer.current_epoch >= self.delay_epochs:
            self._run(trainer, pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.perform_on_test:
            self._run(trainer, pl_module)

    @rank_zero_only
    def _run(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.dataloader is None:
            self._create_dataloader()

        if not trainer.sanity_checking and not trainer.fast_dev_run:
            torch.cuda.empty_cache()
            pl_module.eval()

            self.run(trainer, pl_module)

            torch.cuda.empty_cache()
            pl_module.train()

    @rank_zero_only
    def _create_dataloader(self):
        self.dataloader = self.create_dataloader()

    def create_dataloader(self):
        raise NotImplementedError()

    def run(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Run evaluation loop here
        raise NotImplementedError()


class TestDataCallback(TestCallback):
    def __init__(self,
                 dataset: Union[str, Dataset],
                 batch_size: int = 64,
                 num_workers: int = 2,
                 perform_on_validation: bool = True,
                 perform_on_test: bool = False,
                 delay_epochs: int = 0,
                 **kwargs):
        super().__init__(perform_on_validation=perform_on_validation, perform_on_test=perform_on_test,
                         delay_epochs=delay_epochs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def create_dataloader(self):
        if isinstance(self.dataset, str):
            try:
                from ffcv.loader import Loader, OrderOption
            except ImportError:
                raise ImportError("Please install `ffcv` to use the ffcv dataset.")
            return Loader(self.dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          os_cache=True,
                          distributed=False,
                          drop_last=False,
                          order=OrderOption.SEQUENTIAL)
        else:
            return DataLoader(self.dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=self.num_workers,
                              drop_last=False)


class ZeroShotCallback(TestDataCallback):
    def __init__(self,
                 dataset: Union[str, Dataset],
                 modality_forward=None,
                 text_forward=None,
                 dataset_name: str = None,
                 batch_size: int = 64,
                 num_workers: int = 2,
                 simple_templates: bool = False,
                 top_k: Tuple[int, ...] = (1, 2, 5, 10),
                 accuracy_average: str = 'micro',
                 verbose: bool = False,
                 classes: List = None,
                 multi_label: bool = False,
                 templates: List = None,
                 tokenizer: Optional[Callable] = None,
                 prefix_special_token: Optional[str] = None,
                 perform_on_validation: bool = True,
                 perform_on_test: bool = False,
                 delay_epochs: int = 0,
                 **kwargs):
        super().__init__(perform_on_validation=perform_on_validation, perform_on_test=perform_on_test,
                         delay_epochs=delay_epochs, batch_size=batch_size, num_workers=num_workers, dataset=dataset)
        self.modality_forward = modality_forward
        self.text_forward = text_forward
        self.dataset_name = getattr(dataset, 'name', dataset_name)
        self.top_k = top_k
        self.multi_label = multi_label
        self.accuracy_average = accuracy_average
        self.simple_templates = simple_templates
        self.verbose = verbose
        self.prefix_special_token = prefix_special_token

        if self.dataset_name is None:
            raise ValueError("Dataset name is required for ZeroShotCallback.")

        self.tokenizer = tokenizer
        if self.tokenizer is not None and self.prefix_special_token is not None:
            self.tokenizer.tokenizer.add_special_tokens(['<|visual|>', '<|auditory|>'])

        self.classes = get_classes(self.dataset_name) if classes is None else classes
        self.templates = get_templates(self.dataset_name,
                                       simple=self.simple_templates) if templates is None else templates

    def run_infer(self, pl_module: pl.LightningModule):
        device = pl_module.device
        if self.verbose:
            print("Performing Zero-Shot-Evaluation on dataset: ", self.dataset_name, device)

        classifier = _create_zero_shot_classifier(forward_func=self.text_forward,
                                                  classnames=self.classes,
                                                  templates=self.templates,
                                                  tokenizer=self.tokenizer,
                                                  batch_size=self.batch_size,
                                                  device=device,
                                                  verbose=self.verbose)

        result = _evaluate_zero_shot(forward_func=self.modality_forward,
                                     classifier=classifier,
                                     dataloader=self.dataloader,
                                     top_k=self.top_k,
                                     average=self.accuracy_average,
                                     device=device,
                                     dtype=pl_module.dtype,
                                     multi_label=self.multi_label,
                                     verbose=self.verbose)
        del classifier
        gc.collect()

        return result

    def run(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        result = self.run_infer(pl_module)

        for k, value in result.items():
            if hasattr(trainer.logger, 'log_metrics'):
                trainer.logger.log_metrics({f'zs/{self.dataset_name}_{k}': value}, step=trainer.global_step)
