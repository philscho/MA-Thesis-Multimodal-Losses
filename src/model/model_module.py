import lightning as pl
import torch
from torch import nn
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
)
from transformers import (
    VisionTextDualEncoderProcessor,
)

from ..utils.loss_functions import NTXentLoss
from ..utils.utils import (
    calculate_accuracy,
    get_negative_embeddings,
    calculate_accuracy_simclr,
)
from ..utils.optimizer_and_scheduler import get_optimizer, get_scheduler

# Dimensions of the BERT-Base text encoder hidden state and vocabulary
EMBEDDING_DIM = 768
VOCAB_SIZE = 30522
# Intermediate dimension of the ITM classification head
ITM_HIDDEN_DIM = 512


class LitMML(pl.LightningModule):
    """PyTorch Lightning module for multimodal contrastive learning.

    Trains a dual-stream Vision-Language model (ViT + BERT) with a configurable
    combination of loss functions: CLIP contrastive loss, Image-Text Matching (ITM),
    SimCLR, and Masked Language Modelling (MLM).

    Parameters
    ----------
    model : nn.Module
        A HuggingFace ``VisionTextDualEncoderModel`` (or ``MLMWrapper`` thereof).
    processor : VisionTextDualEncoderProcessor
        Paired image/text processor for the dual-encoder.
    loss_cfg : omegaconf.DictConfig
        Config with a ``losses`` list containing any subset of
        ``["contrastive", "image_text_matching", "SimCLR"]``.
    optimizer_cfg : omegaconf.DictConfig
        Passed to :func:`~src.utils.optimizer_and_scheduler.get_optimizer`.
    scheduler_cfg : omegaconf.DictConfig
        Passed to :func:`~src.utils.optimizer_and_scheduler.get_scheduler`.
    augmentation : callable, optional
        Optional image augmentation applied *inside* the training step so that
        it runs on-GPU and is excluded from validation.
    """

    def __init__(
        self,
        model: nn.Module,
        processor: VisionTextDualEncoderProcessor,
        loss_cfg,
        optimizer_cfg,
        scheduler_cfg,
        augmentation=None,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.loss_cfg = loss_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.model.logit_scale.requires_grad = False
        self.augmentation = augmentation
        self.save_hyperparameters(
            ignore=["model", "processor", "augmentation"]
        )
        self._set_loss_functions(loss_cfg)

    def _set_loss_functions(self, loss_cfg):
        if "contrastive" in self.loss_cfg.losses:
            self.contrastive_loss = ContrastiveLossWithTemperature()

        if "image_text_matching" in self.loss_cfg.losses:
            self.matching_loss = nn.CrossEntropyLoss()
            self.itm_head = nn.Sequential(
                nn.Linear(self.model.config.projection_dim * 2, ITM_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(ITM_HIDDEN_DIM, 2),
            )

        if "SimCLR" in self.loss_cfg.losses:
            self.simclr_loss = NTXentLoss()

    def common_step(self, batch):
        """Run a forward pass and compute all active losses.

        Returns
        -------
        losses : dict[str, Tensor]
            Per-loss scalar tensors, e.g. ``{"loss-contrastive": ..., "loss-simclr": ...}``.
        metrics : dict[str, Tensor]
            Per-loss accuracy scalars for logging.
        """
        token = batch.input_ids
        images = batch.pixel_values
        token_type_ids = batch.token_type_ids
        attention_mask = batch.attention_mask
        if "SimCLR" in self.loss_cfg.losses:
            images_v2 = batch.pixel_values_2
        
        #torch.use_deterministic_algorithms(False)
        #print_memory_usage("After loading batch:")
        if self.augmentation:
            images_v1 = self.augmentation(images.to(torch.uint8))
        else:
            images_v1 = images
        # torch.use_deterministic_algorithms(True)
        # all_images = torch.cat((images_v1, images_v2), dim=0)

        #print_memory_usage("After augmenting images:")
        outputs = self.model(
            pixel_values=images_v1, 
            input_ids=token, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
        #outputs = self.model(**batch)

        #print_memory_usage("After model forward pass:")
        # Ouptut embeddings are already normalized
        image_embeds, text_embeds = outputs.image_embeds, outputs.text_embeds
        # batch_size = images.size(0)
        # image_embeds = outputs.image_embeds[:batch_size]
        # image_embeds_v2 = outputs.image_embeds[batch_size:]
        # text_embeds = outputs.text_embeds

        losses, metrics = {}, {}
        if "contrastive" in self.loss_cfg.losses:
            #loss = clip_contrastive_loss(image_out, text_out, self.loss_cfg.temperature).mean()
            #loss = outputs['loss']
            loss_contrastive = self.contrastive_loss(image_embeds, text_embeds)
            accuracy_contrastive = calculate_accuracy(image_embeds, text_embeds)
            losses["loss-contrastive"] = loss_contrastive
            metrics["acc-contrastive"] = accuracy_contrastive
        
        #print_memory_usage("After calculating contrastive loss:")
        #TODO: put loss in seperate function
        if "image_text_matching" in self.loss_cfg.losses:
            bs = image_embeds.size(0)
            _, neg_text_embeds = get_negative_embeddings(
                image_embeds, text_embeds, outputs.logits_per_image, outputs.logits_per_text)
            selection = torch.randint(0, 2, (bs,)).to(image_embeds.device)
            selected_text_embeds = torch.where(selection.unsqueeze(1) == 0, text_embeds, neg_text_embeds)
            multimodal_embeds = torch.concat((image_embeds, selected_text_embeds), dim=1)
            logits = self.itm_head(multimodal_embeds)
            #probs = F.softmax(logits, dim=1)
            #loss_matching = self.matching_loss(logits, selection.unsqueeze(1).float())
            loss_matching = self.matching_loss(logits, selection)
            preds = logits.argmax(dim=1)
            accuracy_matching = (preds == selection).sum() / len(selection)
            losses['loss-matching'] = loss_matching
            metrics["acc-matching"] = accuracy_matching

            del neg_text_embeds, batch


        del outputs, text_embeds
        torch.cuda.empty_cache()

        #print_memory_usage("After calculating matching loss:")
        if "SimCLR" in self.loss_cfg.losses:
            #images_v2 = self.augmentation(images.to(torch.uint8))
            del images
            image_embeds_v2 = self.model.get_image_features(pixel_values=images_v2)
            del images_v2
            torch.cuda.empty_cache()
            image_embeds_v2 = image_embeds_v2 / image_embeds_v2.norm(dim=-1, keepdim=True) # need to be normalized
            loss_simclr = self.simclr_loss(image_embeds, image_embeds_v2, pl_module=self)
            accuracy_simclr = calculate_accuracy_simclr(self.simclr_loss.logits)
            losses["loss-simclr"] = loss_simclr
            metrics["acc-simclr"] = accuracy_simclr
        elif "SimCLR_v2" in self.loss_cfg.losses:
            #images_v2 = self.augmentation(images.to(torch.uint8))
            del images
            image_embeds_v2 = self.model.vision_model(pixel_values=images_v2, output_hidden_states=True)
            del images_v2
            torch.cuda.empty_cache()
            image_embeds_v2 = image_embeds_v2 / image_embeds_v2.norm(dim=-1, keepdim=True) # need to be normalized
            loss_simclr = self.simclr_loss(image_embeds, image_embeds_v2, pl_module=self)
            accuracy_simclr = calculate_accuracy_simclr(self.simclr_loss.logits)
            losses["loss-simclr"] = loss_simclr
            metrics["acc-simclr"] = accuracy_simclr
        
        #print_memory_usage("After calculating SimCLR loss:")
        return losses, metrics

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        losses, metrics = self.common_step(batch)
        loss = sum(losses.values()) / len(losses)
        self.log("loss-train", loss, sync_dist=True, prog_bar=True)
        self._log_losses_and_metrics(losses, metrics, suffix="train")
        if "contrastive" in self.loss_cfg.losses:
            self.log(f"temperature", self.contrastive_loss.logit_scale)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        losses, metrics = self.common_step(batch)
        loss = sum(losses.values()) / len(losses)
        self.log("loss-val", loss, sync_dist=True, prog_bar=True)
        self._log_losses_and_metrics(losses, metrics, suffix="val")
        return loss
                
    def _log_losses_and_metrics(self, losses, metrics, suffix=""):
        for d in [losses, metrics]:
            for name, val in d.items():
                self.log(f"{name}-{suffix}", val, sync_dist=True)


    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer_cfg, params=self.parameters())
        
        if not self.scheduler_cfg.enabled:
            print("No scheduler provided, using only optimizer")
            return optimizer        
        else:
            if self.scheduler_cfg.name == "ReduceLROnPlateau":
                assert self.scheduler_cfg.monitor is not None
            if self.scheduler_cfg.name == "CosineWarmup":
                num_warmup_steps = self.scheduler_cfg.kwargs.pop("num_warmup_steps")
                num_training_steps = self.scheduler_cfg.kwargs.pop("num_training_steps")
                if num_warmup_steps == "epoch" and num_training_steps == "all":
                    num_warmup_steps = (
                        self.trainer.estimated_stepping_batches / self.trainer.max_epochs
                    )
                    num_training_steps = (
                        self.trainer.estimated_stepping_batches - num_warmup_steps
                    )
                self.scheduler_cfg.kwargs.update({
                    "num_warmup_steps": num_warmup_steps,
                    "num_training_steps": num_training_steps
                })

            monitor_metric = self.scheduler_cfg.pop("monitor")
            interval = self.scheduler_cfg.pop("interval")
            scheduler = get_scheduler(self.scheduler_cfg, optim=optimizer)
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": interval,
                "monitor": monitor_metric,
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }


class MLMWrapper(torch.nn.Module):
    """Wraps a ``VisionTextDualEncoderModel`` with an MLM prediction head.

    Adds a linear projection from BERT's hidden size to the vocabulary so that
    Masked Language Modelling loss can be computed alongside CLIP/ITM losses.
    The wrapper proxies ``config``, ``logit_scale``, and the feature extraction
    methods so that downstream callbacks work transparently.

    Parameters
    ----------
    model : nn.Module
        Base ``VisionTextDualEncoderModel`` to wrap.
    """

    def __init__(self, model):
        super().__init__()
        self.basemodel = model
        self.mlm_head = torch.nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

        # Expose attributes needed by evaluation callbacks
        if hasattr(model, 'config'):
            self.config = model.config
        self.logit_scale = model.logit_scale
        self.get_text_features = model.get_text_features
        self.get_image_features = model.get_image_features

    def forward(self, *args, **kwargs):
        return self.basemodel(*args, **kwargs)

