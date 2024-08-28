from typing import Tuple, List, Mapping, Optional

import math
import numpy as np

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    _gather_embeddings_and_labels,
)

import logging

from lightning.fabric.utilities.rank_zero import rank_prefixed_message, rank_zero_only


def get_custom_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    initial_lr: float
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return current_step / num_warmup_steps * (1 - initial_lr) + initial_lr
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps))))
    
    return LambdaLR(optimizer, lr_lambda)


# ### From transformers.optimization
# def _get_cosine_schedule_with_warmup_lr_lambda(
#     current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
# ):
#     if current_step < num_warmup_steps:
#         return float(current_step) / float(max(1, num_warmup_steps))
#     progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#     return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_subset_of_dataset(dataset, subset_fraction):
    num_samples = int(
        np.floor(len(dataset) * subset_fraction)
    )
    indices = np.random.default_rng(seed=42).choice(
        len(dataset) + 1,
        size=num_samples,
        replace=False,
    )
    print(f"Using a {subset_fraction} subset of {dataset.__class__.__name__}")

    return torch.utils.data.Subset(dataset, indices=indices)


def calculate_accuracy(images, texts):
    # all_gpus: global_batch_size, embedding_dim), labels: (local_batch_size)
    (images_all_gpus, texts_all_gpus, labels) = _gather_embeddings_and_labels(images, texts)
    # shape: (local_batch_size, global_batch_size)
    logits_per_image = torch.matmul(images, texts_all_gpus.transpose(0, 1))
    logits_per_text = torch.matmul(texts, images_all_gpus.transpose(0, 1))
    acc_per_image = (logits_per_image.argmax(dim=-1) == labels).sum()
    acc_per_text = (logits_per_text.argmax(dim=-1) == labels).sum()
    accuracy = (acc_per_image + acc_per_text) / 2 / logits_per_image.size(0)
    return accuracy


def get_negative_embeddings(
    image_embeds: Tensor,
    text_embeds: Tensor,
    similarity_i2t: Tensor,
    similarity_t2i: Tensor,
    text_atts: Tensor = None
) -> Tuple[Tensor, Tensor, Tensor]:
    with torch.no_grad():
        bs = image_embeds.size(0)
        weights_i2t = F.softmax(similarity_i2t[:, :bs], dim=1)
        weights_t2i = F.softmax(similarity_t2i[:, :bs], dim=1)
        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)

    image_embeds_neg, text_embeds_neg, text_atts_neg = [], [], []
    for b in range(bs):
        neg_idx = int(torch.multinomial(weights_t2i[b], 1).item())
        image_embeds_neg.append(image_embeds[neg_idx])
    image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

    for b in range(bs):
        neg_idx = int(torch.multinomial(weights_i2t[b], 1).item())
        text_embeds_neg.append(text_embeds[neg_idx])
        #text_atts_neg.append(text_atts[neg_idx])   #TODO: implement text attention mask?
    text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
    #text_atts_neg = torch.stack(text_atts_neg, dim=0)
    return image_embeds_neg, text_embeds_neg #, text_atts_neg


def create_views(
        image_batch,
        augmentation,
        n=2
) -> Tuple[Tensor]:
    z1, z2 = augmentation(image_batch), augmentation(image_batch)
    return z1, z2


class RankedLogger(logging.LoggerAdapter):
    """ A multi-device friendly logger that prefixes messages with the rank. """
    def __init__(
            self,
            name: str = __name__,
            rank_zero_only: bool = False,
            extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """
        Initialize a new logger.

        Parameters
        ----------
        name: str = __name__
            The name of the logger
        rank_zero_only: bool = False
            If True, only log on the main process (rank 0)
        extra: Optional[Mapping[str, object]] = None
            Extra information to be added to the log message.
        """
        super().__init__(logging.getLogger(name), extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                self.logger.log(level, msg, *args, **kwargs)