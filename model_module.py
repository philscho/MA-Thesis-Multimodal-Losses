# import all of the modules needed
import lightning as pl
import torch
from torch import nn
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
)
from transformers import (
    VisionTextDualEncoderProcessor,
)

from utils.loss_functions import NTXentLoss
from utils.utils import (
    calculate_accuracy,
    get_negative_embeddings, #print_memory_usage,
    get_negative_embeddings, #print_memory_usage,
)
from utils.optimizer_and_scheduler import get_optimizer, get_scheduler

class LitMML(pl.LightningModule):
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
        # self.validation_step_outputs = []
        # self.temperature = temperature
        # self.learning_rate = learning_rate
        self.augmentation = augmentation
        self.save_hyperparameters(
            ignore=["model", "processor", "augmentation"]
        )
        self._set_loss_functions(loss_cfg)

    def _set_loss_functions(self, loss_cfg):
        self.contrastive_loss = ContrastiveLossWithTemperature()

        if "image_text_matching" in self.loss_cfg.losses:
            self.matching_loss = nn.CrossEntropyLoss()
            self.itm_head = nn.Sequential(
                nn.Linear(
                    self.model.config.projection_dim * 2, 512
                ),  # TODO: make dims variable
                nn.ReLU(),
                nn.Linear(512, 2),
            )
        if "SimCLR" in self.loss_cfg.losses:
            self.simclr_loss = NTXentLoss()

    def common_step(self, batch):
        #print_memory_usage("Beginning of step:")
        token = batch.input_ids
        images = batch.pixel_values
        token_type_ids = batch.token_type_ids
        attention_mask = batch.attention_mask
        
        #torch.use_deterministic_algorithms(False)
        #print_memory_usage("After loading batch:")
        images_v1 = self.augmentation(images.to(torch.uint8))
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
            loss_matching = self.matching_loss(logits, selection.long())
            preds = logits.argmax(dim=1)
            accuracy_matching = (preds == selection).sum() / len(selection)
            losses['loss-matching'] = loss_matching
            metrics["acc-matching"] = accuracy_matching

            del neg_text_embeds, batch


        del outputs, text_embeds
        torch.cuda.empty_cache()

        #print_memory_usage("After calculating matching loss:")
        if "SimCLR" in self.loss_cfg.losses:
            images_v2 = self.augmentation(images.to(torch.uint8))
            del images
            image_embeds_v2 = self.model.get_image_features(pixel_values=images_v2)
            del images_v2
            torch.cuda.empty_cache()
            image_embeds_v2 = image_embeds_v2 / image_embeds_v2.norm(dim=-1, keepdim=True) # need to be normalized
            loss_simclr = self.simclr_loss(image_embeds, image_embeds_v2, pl_module=self)
            #accuracy_simclr = calculate_accuracy(z_1, z_2)
            losses["loss-simclr"] = loss_simclr
            #metrics["acc-simclr"] = accuracy_simclr
        
        #print_memory_usage("After calculating SimCLR loss:")
        return losses, metrics

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        losses, metrics = self.common_step(batch)
        loss = sum(losses.values()) / len(losses)
        self.log("loss-train", loss, sync_dist=True, prog_bar=True)
        self.log(f"temperature", self.contrastive_loss.logit_scale)
        self._log_losses_and_metrics(losses, metrics, suffix="train")
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


    # def on_validation_epoch_start(self):
    #     self.cifar10_classifier = _create_zero_shot_classifier(
    #         forward_func=lambda x: self.model.get_text_features(input_ids=x),
    #         classnames=[
    #             "airplane",
    #             "automobile",
    #             "bird",
    #             "cat",
    #             "deer",
    #             "dog",
    #             "frog",
    #             "horse",
    #             "ship",
    #             "truck",
    #         ],
    #         templates="a photo of a {}",
    #         tokenizer=self.processor,
    #     )
    #     self.caltech101_classifier = _create_zero_shot_classifier(
    #         forward_func=lambda x: self.model.get_text_features(input_ids=x),
    #         classnames=self.trainer.val_dataloaders[2].dataset.categories,
    #         templates="a photo of a {}",
    #         tokenizer=self.processor,
    #     )

    # def on_validation_epoch_end(self):
    #     cifar_10_result = _evaluate_zero_shot(
    #         forward_func=lambda x: self.model.get_image_features(pixel_values=x),
    #         classifier=self.cifar10_classifier,
    #         dataloader=self.trainer.val_dataloaders[1],
    #         confusion_matrix=True,
    #         top_k=(1,),
    #     )
    #     caltech_101_result = _evaluate_zero_shot(
    #         forward_func=lambda x: self.model.get_image_features(pixel_values=x),
    #         classifier=self.caltech101_classifier,
    #         dataloader=self.trainer.val_dataloaders[2],
    #         confusion_matrix=True,
    #         top_k=(1,),
    #     )

    #     # TODO: refactor this
    #     for k, v in cifar_10_result.items():
    #         if k == "ConfusionMatrix":
    #             self.logger.log_image(
    #                 key="cifar-10-confusionmatrix", images=[v], caption=["ConfMatrix"]
    #             )
    #         else:
    #             self.log("cifar10-accuracy", v, sync_dist=False)
    #     for k, v in caltech_101_result.items():
    #         if k == "ConfusionMatrix":
    #             self.logger.log_image(
    #                 key="caltech-101-confusionmatrix",
    #                 images=[v],
    #                 caption=["ConfMatrix"],
    #             )
    #         else:
    #             self.log("caltech101-accuracy", v, sync_dist=False)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer_cfg, params=self.parameters())
        
        if self.scheduler_cfg.name is None:
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
