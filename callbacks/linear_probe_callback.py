# %%
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
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class LinearProbeCallback(pl.Callback):

    def __init__(
        self,
        val_dataloader_idx: int,
        linear_probe: torch.nn.Linear,
        log_str_prefix: str = "unnameddataset",  # do not include 'linear-probe' in the prefix
        logging_interval: str = "epoch",
        log_every: int = 1,
        confusion_matrix: bool = True,
        top_k: tuple = (1,),
        max_epochs: int = 100,
        tolerance: float = 0.0001,
        verbose: bool = False,
        optim: torch.optim = torch.optim.Adam,
        lossfn: str = "crossentropy",
    ) -> None:
        super().__init__()

        self.logging_interval = logging_interval

        if logging_interval not in (None, "step", "epoch"):
            raise MisconfigurationException(
                "logging_interval should be `step` or `epoch` or `None`."
            )
        if logging_interval == "step":
            raise NotImplementedError("Not implemented at the step level yet")

        self.val_dataloader_idx = val_dataloader_idx
        self.log_every = (
            log_every  # currently takes only the dataset name due to legacy issues
        )
        self.confusion_matrix = confusion_matrix
        self.top_k = top_k
        self.log_str_prefix = log_str_prefix

        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.verbose = verbose
        self.optim = optim
        self.lossfn = lossfn
        self.linear_probe = linear_probe

    def run_now_flag(self, trainer: pl.Trainer):
        if self.logging_interval == "epoch":
            run_now_flag = (trainer.current_epoch + 1) % self.log_every == 0
        return run_now_flag

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:

        if self.run_now_flag(trainer):
            if trainer.is_global_zero:
                self.trained_linear_probe = create_linear_probe(
                    forward_func=lambda x: pl_module.model.get_image_features(
                        pixel_values=x
                    ),
                    dataloader=trainer.val_dataloaders[self.val_dataloader_idx],
                    linear_layer=self.linear_probe,
                    max_epochs=self.max_epochs,
                    tolerance=self.tolerance,
                    local_optimizer=self.optim,
                    verbose=self.verbose,
                )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:

        if self.run_now_flag(trainer):  # run only on some intervals
            if trainer.is_global_zero:  # run only on rank 0
                result = eval_linear_probe(
                    forward_func=lambda x: pl_module.model.get_image_features(
                        pixel_values=x
                    ),
                    classifier=self.trained_linear_probe,
                    dataloader=trainer.val_dataloaders[self.val_dataloader_idx],
                    confusion_matrix=self.confusion_matrix,
                    top_k=self.top_k,
                    verbose=self.verbose,
                )

                # assert len(result) == 1
                for k, v in result.items():
                    if k == "ConfusionMatrix":
                        trainer.logger.log_image(
                            key=f"linear-probe-{self.log_str_prefix}-confusionmatrix",
                            images=[v],
                            caption=["ConfMatrix"],
                        )
                    else:
                        self.log(
                            f"{self.log_str_prefix}-linear-probe-accuracy",
                            v,
                            sync_dist=False,
                        )


def create_linear_probe(
    forward_func: callable,
    dataloader: torch.utils.data.DataLoader,
    linear_layer: Union[torch.utils.data.DataLoader, pl.LightningDataModule],
    device: str = "cuda",
    max_epochs: int = 50,
    tolerance: float = 0.0001,
    local_optimizer: torch.optim = torch.optim.Adam,
    local_optimizer_kwargs: dict = {"lr": 0.0001},
    verbose: bool = False,
    local_lossfn: str = "crossentropy",
):
    """
    Creates a linear probe that is trained on the dataloader for crossentropy

    Args:
        forward_func : the function to get features from the networks
        dataloader : dataloader for the images
        linear_layer : the linear layer that is to be trained
        device : device on which to run the training
        max_epochs : number of epochs to finetune the linear probe
        tolerance : threshold at which if the change in loss is not observed, early stopping is triggered. we right now wait for 5 epochs
        local_lossfn : the loss on which to train the linear layer

    Returns:
        The trained linear layer
    """

    features_dict = []
    for images, labels in tqdm(dataloader, desc="Getting all the features"):
        feats = forward_func(images.to(device))
        features_dict.append((feats.detach().cpu(), labels.cpu()))

    # add linear layer
    # linear_layer = torch.nn.Linear(linear_layer_in_features, num_classes)
    linear_layer.requires_grad = True
    linear_layer.train()
    linear_layer.to(device)

    # train the linear layer
    localoptim = local_optimizer(linear_layer.parameters(), **local_optimizer_kwargs)

    num_times_loss_not_changing = 0
    prev_loss = 0.0

    losses_across_dataset = []
    with torch.enable_grad():
        epoch_bar = tqdm(range(max_epochs))
        for epoch in epoch_bar:
            epoch_bar.set_description(f"Epoch:{epoch}")

            loss_across_dataset = 0.0
            for _, batch in enumerate(features_dict):

                features, labels = batch
                localoptim.zero_grad()
                out = linear_layer(features.to(device))

                if local_lossfn == "crossentropy":
                    loss = torch.nn.functional.cross_entropy(out, labels.to(device))
                else:
                    raise NotImplementedError(
                        "If you want to use regression methods, you gotta implement it..."
                    )

                loss_across_dataset += loss
                loss.backward()
                localoptim.step()

            loss_across_dataset = loss_across_dataset / len(dataloader)
            losses_across_dataset.append(loss_across_dataset.item())

            lossdiff = torch.abs(prev_loss - loss_across_dataset)
            if lossdiff < tolerance:  # same value as sklearn's logistic regression
                num_times_loss_not_changing += 1

            epoch_bar.set_postfix(
                {
                    "loss": loss_across_dataset.item(),
                    "diff": lossdiff.item(),
                }
            )

            prev_loss = loss_across_dataset.item()

            # two break conditions
            if num_times_loss_not_changing == 5:
                print(
                    f"CONVERGENCE!! Quitting with loss {loss_across_dataset.item()} after running {epoch} epochs...\n"
                )
                break

    if verbose:
        import time  # lazy import
        import matplotlib.pyplot as plt  # lazy import

        plt.figure()
        plt.plot(losses_across_dataset)
        plt.title("Local loss vs Epochs for training the linear layer")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.legend()
        plt.savefig(
            f"log_local_training_{time.strftime('%a, %d %b %Y %H:%M:%S ',time.localtime())}.png"
        )
        plt.close()
    return linear_layer.eval()


def eval_linear_probe(
    forward_func: callable,
    classifier: any,
    dataloader: Union[torch.utils.data.DataLoader, pl.LightningDataModule],
    top_k: Tuple[int, ...] = (1, 2, 5, 10),
    average: str = "micro",
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float32,
    confusion_matrix: bool = False,
    multi_label: bool = False,
    verbose: bool = False,
):
    """
    Evaluates the linear probe for accuracy and confusion matrix on the dataloader

    Returns:
        Torchmetrics result instance
    """

    metric_kwargs = dict(dist_sync_on_step=False, sync_on_compute=False)

    num_classes = 10  # this better be 10 for cifar1

    if multi_label:
        metric = {
            f"mAP": MultilabelAveragePrecision(
                num_labels=num_classes, average="macro", **metric_kwargs
            ),
        }
    else:
        metric = {
            f"Top{k}Accuracy": MulticlassAccuracy(
                top_k=k, average=average, num_classes=num_classes, **metric_kwargs
            )
            for k in top_k
        }

    if confusion_matrix:
        metric["ConfusionMatrix"] = MulticlassConfusionMatrix(
            num_classes=num_classes, normalize=None, **metric_kwargs
        )
    metric = MetricCollection(metric).to(device)

    classifier = classifier.to(dtype=dtype).eval()
    with torch.no_grad():
        bar = (
            tqdm(dataloader, desc=f"Predicting...", total=len(dataloader))
            if verbose
            else dataloader
        )
        for point in bar:
            inputs, target = point
            inputs = inputs.to(device)
            target = target.to(device)

            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                features = forward_func(inputs)

            logits = classifier(features)

            step_metric = metric(logits, target.squeeze().long())

            if verbose and average is not None:
                bar.set_postfix(
                    {
                        k: v.item()
                        for k, v in step_metric.items()
                        if k != "ConfusionMatrix"
                    }
                )

    result = metric.compute()
    if verbose:
        print(f"ZS result {result}")

    for k, v in result.items():
        result[k] = v.cpu().numpy().item() if v.dim() == 0 else v.cpu().numpy()
    return result


# this should allow inheriting something like this...

# class CIFAR10LinearProbe(LinearProbeCallback):
#     def __init__(
#         self,
#         val_dataloader_idx: int,
#         linear_probe: Linear,
#         log_str_prefix: str = "cifar10",
#         logging_interval: str = "epoch",
#         log_every: int = 1,
#         confusion_matrix: bool = True,
#         top_k: Tuple = (1,),
#         max_epochs: int = 100,
#         tolerance: float = 0.0001,
#         verbose: bool = False,
#         optim: torch.optim.Optimizer = torch.optim.Adam,
#         lossfn: str = "crossentropy",
#     ) -> None:
#         super().__init__(
#             val_dataloader_idx,
#             linear_probe,
#             log_str_prefix,
#             logging_interval,
#             log_every,
#             confusion_matrix,
#             top_k,
#             max_epochs,
#             tolerance,
#             verbose,
#             optim,
#             lossfn,
#         )
