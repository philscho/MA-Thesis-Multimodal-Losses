# %%
from typing import Any, Tuple, Union, List, Dict

import torch
from torch import nn
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
        train_dataloader: torch.utils.data.dataloader.DataLoader,
        test_dataloader: torch.utils.data.dataloader.DataLoader,
        input_dim: int,
        num_classes: int,
        dataset_name: str = "unnameddataset",  # do not include 'linear-probe' in the prefix
        logging_interval: str = "epoch",
        log_every: int = 1,
        confusion_matrix: bool = True,
        top_k: tuple = (1,),
        max_epochs: int = 100,
        tolerance: float = 0.0001,
        verbose: bool = False,
        optim: torch.optim = torch.optim.SGD,
        lossfn: str = "crossentropy",
        device: Union[str, torch.device] = "cuda",
        learning_rates: List[float] = [0.0001],
    ) -> None:
        super().__init__()

        self.logging_interval = logging_interval

        if logging_interval not in (None, "step", "epoch"):
            raise MisconfigurationException(
                "logging_interval should be `step` or `epoch` or `None`."
            )
        if logging_interval == "step":
            raise NotImplementedError("Not implemented at the step level yet")

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.log_every = (
            log_every  # currently takes only the dataset name due to legacy issues
        )
        self.confusion_matrix = confusion_matrix
        self.top_k = top_k
        self.dataset_name = dataset_name

        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.verbose = verbose
        self.optim = optim
        self.lossfn = lossfn
        self.device = device
        self.learning_rates = learning_rates

    def run_now_flag(self, trainer: pl.Trainer):
        if self.logging_interval == "epoch":
            run_now_flag = (trainer.current_epoch + 1) % self.log_every == 0
        return run_now_flag

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        
        if self.run_now_flag(trainer):
            if trainer.is_global_zero:
                print(f"Starting linear probe training for {self.dataset_name}...")
                classifiers, params = setup_linear_classifiers(
                    out_dim=self.input_dim,
                    num_classes=self.num_classes,
                    learning_rates=self.learning_rates,
                )
                self.trained_classifiers = train_linear_probe(
                    # forward_func=lambda x: pl_module.model.get_image_features(
                    #     pixel_values=x
                    # ),
                    # forward_func=lambda x: pl_module.model.vision_model(
                    #     pixel_values=x
                    # ).pooler_output,
                    # forward_func=lambda x: pl_module.model.basemodel.vision_model(
                    #     pixel_values=x
                    # ).pooler_output,
                    forward_func=lambda x: pl_module.model.basemodel.vision_model( #forward_func=lambda x: pl_module.model.vision_model(
                        pixel_values=x
                    ).last_hidden_state[:, 0, :], # CLS token
                    dataloader=self.train_dataloader,
                    classifiers=classifiers,
                    parameters=params,
                    max_epochs=self.max_epochs,
                    tolerance=self.tolerance,
                    local_optimizer=self.optim,
                    verbose=self.verbose,
                    device=self.device,
                    logger = trainer.logger,
                    dataset_name=self.dataset_name,
                )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:

        if self.run_now_flag(trainer):  # run only on some intervals
            if trainer.is_global_zero:  # run only on rank 0
                self.result_per_classifier = eval_linear_probe(
                    # forward_func=lambda x: pl_module.model.get_image_features(
                    #     pixel_values=x
                    # ),
                    # forward_func=lambda x: pl_module.model.vision_model(
                    #     pixel_values=x
                    # ).pooler_output,
                    # forward_func=lambda x: pl_module.model.basemodel.vision_model(
                    #     pixel_values=x
                    # ).pooler_output,
                    forward_func=lambda x: pl_module.model.basemodel.vision_model( #forward_func=lambda x: pl_module.model.vision_model(
                        pixel_values=x
                    ).last_hidden_state[:, 0, :], # CLS token
                    classifiers=self.trained_classifiers,
                    dataloader=self.test_dataloader,
                    num_classes=self.num_classes,
                    confusion_matrix=self.confusion_matrix,
                    top_k=self.top_k,
                    verbose=self.verbose,
                    device=self.device,
                )

                # assert len(result) == 1
                self.result = None
                best_accuracy = 0.0
                for classifier, result in self.result_per_classifier.items():
                    for k, v in result.items():
                        if k == "ConfusionMatrix":
                            trainer.logger.log_image(
                                key=f"{self.dataset_name}-linear-probe-{classifier}-confusionmatrix",
                                images=[v],
                                caption=["ConfMatrix"],
                            )
                        else:
                            trainer.logger.log_metrics(
                                {f"{self.dataset_name}-linear-probe-{classifier}-{k}": v},
                                #sync_dist=False,
                            )
                        if k == "Top1Accuracy" and v > best_accuracy:
                            best_accuracy = v
                            self.result = result


def train_linear_probe(
    forward_func: callable,
    dataloader: torch.utils.data.DataLoader,
    classifiers: 'AllClassifiers',
    parameters: List[Dict[str, Any]],
    dataset_name,
    device: str = "cuda",
    logger = None,
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

    classifiers.requires_grad = True
    classifiers.train()
    classifiers.to(device)
    # for k, v in classifiers.classifier_dict.items():
    #     v.requires_grad = True
    #     v.train()
    #     v.to(device)

    # train the linear layer
    localoptim = local_optimizer(parameters, **local_optimizer_kwargs)

    num_times_loss_not_changing = 0
    prev_loss = 0.0

    losses_across_dataset = []
    with torch.enable_grad():
        epoch_bar = tqdm(range(max_epochs))
        for epoch in epoch_bar:
            epoch_bar.set_description(f"Epoch:{epoch}")

            loss_across_dataset = 0.0
            loss_per_probe = {k: 0.0 for k in classifiers.classifiers_dict.keys()}
            for _, batch in enumerate(features_dict):
                features, labels = batch
                localoptim.zero_grad()
                out = classifiers(features.to(device))
                
                total_loss = 0
                # iterate over different classifiers
                for k, v in out.items():
                    if local_lossfn == "crossentropy":
                        loss = torch.nn.functional.cross_entropy(v, labels.to(device))
                        loss_per_probe[k] += loss
                        # logger.log_metrics(
                        #         {f"{dataset_name}-linear-probe-{k}": loss},
                        #         #sync_dist=False,
                        #     )

                    else:
                        raise NotImplementedError(
                            "If you want to use regression methods, you gotta implement it..."
                        )
                    total_loss += loss
                    # here: log(k, v)
                    # calc. accuracy for this classifier

                loss_across_dataset += loss
                total_loss.backward()
                localoptim.step()

            loss_per_probe = {k: v / len(features_dict) for k, v in loss_per_probe.items()}
            if logger:
                logger.log_metrics(
                    {f"{dataset_name}-linear-probe-{k}": loss
                    for k, loss in loss_per_probe.items()}
                    #sync_dist=False,
                )

            # loss_across_dataset = loss_across_dataset / len(dataloader)
            # losses_across_dataset.append(loss_across_dataset.item())

            # lossdiff = torch.abs(prev_loss - loss_across_dataset)
            # if lossdiff < tolerance:  # same value as sklearn's logistic regression
            #     num_times_loss_not_changing += 1

            # epoch_bar.set_postfix(
            #     {
            #         "loss": loss_across_dataset.item(),
            #         "diff": lossdiff.item(),
            #     }
            # )

            # prev_loss = loss_across_dataset.item()

            # # two break conditions
            # if num_times_loss_not_changing == 5:
            #     print(
            #         f"CONVERGENCE!! Quitting with loss {loss_across_dataset.item()} after running {epoch} epochs...\n"
            #     )
            #     break

    # if verbose:
    #     import time  # lazy import
    #     import matplotlib.pyplot as plt  # lazy import

    #     plt.figure()
    #     plt.plot(losses_across_dataset)
    #     plt.title("Local loss vs Epochs for training the linear layer")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Losses")
    #     plt.legend()
    #     plt.savefig(
    #         f"log_local_training_{time.strftime('%a, %d %b %Y %H:%M:%S ',time.localtime())}.png"
    #     )
    #     plt.close()
    return classifiers.eval()


def eval_linear_probe(
    forward_func: callable,
    classifiers: any,
    dataloader: Union[torch.utils.data.DataLoader, pl.LightningDataModule],
    num_classes: int,
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
    metric_per_classifier = {}
    for k in classifiers.classifiers_dict.keys():
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
        metric_per_classifier[k] = MetricCollection(metric).to(device)

    classifiers = classifiers.to(dtype=dtype).eval()
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

            logits = classifiers(features)
            for k, v in logits.items():
                step_metric = metric_per_classifier[k](v, target.squeeze().long())

            # if verbose and average is not None:
            #     bar.set_postfix(
            #         {
            #             k: v.item()
            #             for k, v in step_metric.items()
            #             if k != "ConfusionMatrix"
            #         }
            #     )

    result_dict = {}
    for classifier, metric in metric_per_classifier.items():
        result = metric.compute()
        if verbose:
            print(f"Linear {classifier} result {result}")
        for k, v in result.items():
            result[k] = v.cpu().numpy().item() if v.dim() == 0 else v.cpu().numpy()
        result_dict[classifier] = result
    
    return result_dict


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self,
                 out_dim: int,
                 num_classes: int = 1000):
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, X):
        return self.linear(X)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


def setup_linear_classifiers(
        out_dim: int,
        num_classes: int,
        learning_rates: List[float],
) -> Union[AllClassifiers, List[Dict[str, Any]]]:
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for lr in learning_rates:
        linear_classifier = LinearClassifier(
            out_dim, num_classes=num_classes
        )

        linear_classifiers_dict[
            f"classifier_{lr:.5f}".replace(".", "_")
        ] = linear_classifier

        optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)

    return linear_classifiers, optim_param_groups


def scale_lr(learning_rates: float, batch_size: int, devices: int) -> float:
    return learning_rates * (batch_size * devices) / 256.0
