from typing import Tuple, Union, List, Dict

import torch
from torch.nn.modules import Linear
from torch.optim.adam import Adam
import torch.utils
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import time
import lightning as pl

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MultilabelAveragePrecision,
    MulticlassConfusionMatrix,
)


class ZeroShotCallback(pl.Callback):
    def __init__(
            self,
            dataset_name: str,
            dataloader: DataLoader,
            classnames: List[str],
            templates: List[List[str]],  # Now a list of template sets
            tokenizer=None,
            text_forward=None,
            modality_forward=None,
            itm_head: torch.nn.Module = None,
            top_k_preds: int = 20,
            batch_size: int = 64,
            device: Union[str, torch.device] = "cuda",
            top_k: Tuple[int, ...] = (1, 2, 5, 10),
            average: str = "micro",
            dtype: torch.dtype = torch.float32,
            confusion_matrix: bool = False,
            multi_label: bool = False,
            verbose: bool = False,
            only_itm: bool = False,
        ) -> None:
        super().__init__()
        self.dataloader = dataloader
        self.dataset_name = dataset_name
        self.classnames = classnames
        self.templates = templates  # List of template sets
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
        self.result = None
        self.itm_head = itm_head
        self.top_k_preds = top_k_preds
        self.only_itm = only_itm

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        print(f"Starting zero-shot evaluation for {self.dataset_name}")

        # Create all classifiers for each template set
        classifiers = []
        template_counts = []
        for template_set in self.templates:
            classifier = _create_zero_shot_classifier(
                forward_func=self.text_forward,
                classnames=self.classnames,
                templates=template_set,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                device=self.device,
                verbose=self.verbose,
            )
            classifiers.append(classifier)
            template_counts.append(len(template_set))

        # Evaluate all classifiers in one pass
        self.result = _evaluate_multi_zero_shot(
            forward_func=self.modality_forward,
            classifiers=classifiers,
            dataloader=self.dataloader,
            itm_head=self.itm_head,
            top_k_preds=self.top_k_preds,
            confusion_matrix=self.confusion_matrix,
            top_k=self.top_k,
            average=self.average,
            multi_label=self.multi_label,
            device=self.device,
            dtype=self.dtype,
            verbose=self.verbose,
            template_counts=template_counts,
            only_itm=self.only_itm,
        )

        # Prepare a list to store results for later saving
        if not hasattr(self, "logged_results"):
            self.logged_results = []

        # Logging and storing
        for idx, template_count in enumerate(template_counts):
            for mode in ["regular", "itm"]:
                if mode not in self.result[idx]:
                    continue
                metrics_dict = {}
                # Add all metrics except ConfusionMatrix to metrics_dict
                for k, v in self.result[idx][mode].items():
                    if k != "ConfusionMatrix":
                        v_to_save = v
                        metrics_dict[k] = v_to_save
                # Trainer logging (keep as before, including ConfusionMatrix)
                for k, v in self.result[idx][mode].items():
                    metric_name = f"{self.dataset_name}-zeroshot-{mode}-{k}-{template_count}_templates"
                    if k == "ConfusionMatrix":
                        trainer.logger.log_image(
                            key=f"{metric_name}-confusionmatrix", images=[v], caption=[f"ConfMatrix-{template_count}-{mode}"]
                        )
                    else:
                        trainer.logger.log_metrics({metric_name: v})
                # Store for later (without ConfusionMatrix)
                self.logged_results.append({
                    "dataset": self.dataset_name,
                    "template_count": template_count,
                    "mode": mode,
                    "metrics": metrics_dict
                })


def _create_zero_shot_classifier(forward_func,
                                 classnames: List[str],
                                 templates: List = None,
                                 tokenizer=None,
                                 batch_size: int = 64,
                                 device: Union[str, torch.device] = "cuda",
                                 verbose: bool = False):
    templates = ['{}'] if templates is None else templates  # templates = ['a photo of a {}.']
    if isinstance(templates, str):
        templates = [templates]
    num_templates = len(templates)  # num_templates = 1
    batch_size = 2 ** ((batch_size // num_templates) - 1).bit_length() # batch_size = 2 ** ((64 // 1) - 1).bit_length() = 2 ** 6 = 64

    batch_class_names = make_batches(classnames, batch_size)
    # if verbose:
    #     print ('batch_class_names : \n ',batch_class_names)
    with torch.no_grad():
        zeroshot_weights = []
        bar = tqdm(batch_class_names, desc="Classifier weights...") if verbose else batch_class_names
        for batch_class_name in bar:
            texts = [template.format(classname) for classname in batch_class_name for template in templates]
            # if verbose:
            #     print ('texts',texts)
            if tokenizer is not None:
                #texts = tokenizer(texts).to(device)  # tokenize Shape: batch_size * num_tokens x context_length
                input = tokenizer(text=texts, padding=True, truncation=True, return_tensors="pt")
                texts = input['input_ids'].to(device)  # tokenize Shape: batch_size * num_tokens x context_length
                mask = input['attention_mask'].to(device)

            class_embeddings = forward_func(texts, mask)  # batch_size * num_tokens x embedding_dim
            #class_embeddings = forward_func(texts)  # batch_size * num_tokens x embedding_dim
            #class_embeddings = forward_func(input_ids=texts)  # batch_size * num_tokens x embedding_dim
            # forward_func(texts) => forward_func(input_ids=texts)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            class_embeddings = class_embeddings.view(len(batch_class_name), num_templates,
                                                     -1)  # batch_size x num_tokens x embedding_dim
            class_embedding_mean = class_embeddings.mean(dim=1)  # batch_size x embedding_dim
            class_embedding_mean /= class_embedding_mean.norm(dim=1).view(-1, 1)

            zeroshot_weights.append(class_embedding_mean)
        zeroshot_weights = torch.concat(zeroshot_weights, dim=0).T
    return zeroshot_weights.to(device)


def _evaluate_zero_shot(
    forward_func,
    classifier,
    dataloader,
    itm_head: torch.nn.Module = None,
    top_k_preds: int = 20,
    top_k: Tuple[int, ...] = (1, 2, 5, 10),
    average: str = "micro",
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float32,
    confusion_matrix: bool = False,
    multi_label: bool = False,
    verbose: bool = False,
) -> Dict:
    num_classes = classifier.shape[1]
    metric_kwargs = dict(dist_sync_on_step=False, sync_on_compute=False)
    if multi_label:
        metric = {
            f'mAP': MultilabelAveragePrecision(num_labels=num_classes, average='macro', **metric_kwargs),
        }
    else:
        metric = {
            f'Top{k}Accuracy': MulticlassAccuracy(top_k=k, average=average, num_classes=num_classes, **metric_kwargs)
            for k in top_k
        }
    if confusion_matrix:
        metric['ConfusionMatrix'] = MulticlassConfusionMatrix(num_classes=num_classes, normalize=None, **metric_kwargs)
    metric = MetricCollection(metric).to(device)
    # metric_itm = MetricCollection(metric.clone()).to(device) if itm_head else None
    metric_itm = MetricCollection({k: v.clone() for k, v in metric.items()}).to(device) if itm_head else None

    classifier = classifier.to(dtype=dtype)
    start_time = time.time()
    with torch.no_grad():
        bar = tqdm(dataloader, desc=f'Predicting...', total=len(dataloader)) if verbose else dataloader
        for point in bar:
            inputs, target = point
            inputs = inputs.to(device)
            target = target.to(device)
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                features = forward_func(inputs)
            features /= features.norm(dim=-1, keepdim=True)
            logits_regular = features @ classifier
            step_metric = metric(logits_regular, target.squeeze().long())
            if itm_head:
                logits_itm = _get_itm_head_predictions(itm_head, logits_regular.clone(), top_k_preds, features, classifier)
                step_metric_itm = metric_itm(logits_itm, target.squeeze().long())
            if verbose and average is not None:
                postfix = {k: v.item() for k, v in step_metric.items() if k != 'ConfusionMatrix'}
                if itm_head:
                    postfix.update({f"itm_{k}": v.item() for k, v in step_metric_itm.items() if k != 'ConfusionMatrix'})
                bar.set_postfix(postfix)
    end_time = time.time()
    print(f"Predictions took {end_time - start_time:.4f} seconds")

    result = {"regular": metric.compute()}
    if itm_head:
        result["itm"] = metric_itm.compute()
    if verbose:
        print(f"ZS result {result}")
    for key in result:
        for k, v in result[key].items():
            result[key][k] = v.cpu().numpy().item() if v.dim() == 0 else v.cpu().numpy()
    return result


def make_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def _get_itm_head_predictions(itm_head, sim_i2t, top_k, image_features, text_features):
    batch_size = sim_i2t.shape[0]
    if sim_i2t.size(1) < top_k:
        top_k = sim_i2t.size(1)
    _, top_indices = torch.topk(sim_i2t, top_k, dim=1)
    top_indices, _ = torch.sort(top_indices, dim=1)
    topk_class_embeds = torch.stack(
        [torch.index_select(text_features, dim=1, index=top_indices[i, :]).T for i in range(batch_size)]
    )
    multimodal_embeddings = torch.cat(
        [image_features.unsqueeze(1).expand(-1, top_k, -1), topk_class_embeds], dim=2
    ).to(torch.float32) # batch_size x top_k x (embedding_dim * 2)
    logits = itm_head(multimodal_embeddings)
    matching_probs = torch.softmax(logits, dim=-1)[:, :, 0]
    mask = torch.full_like(sim_i2t, fill_value=False, dtype=torch.bool)
    for i in range(sim_i2t.size(0)):
        mask[i, top_indices[i]] = True
    sim_i2t.fill_(-float('inf'))
    sim_i2t[mask] = matching_probs.flatten()
    return sim_i2t

def _get_itm_head_predictions_old(itm_head, sim_f2t, top_k, features, classifier):
    batch_size = sim_f2t.shape[0]
    if sim_f2t.size(-1) < top_k:
        top_k = sim_f2t.size(-1)
    _, top_indices = torch.topk(sim_f2t, top_k, dim=-1)
    topk_class_embeds = torch.stack(
        [torch.index_select(classifier, dim=1, index=top_indices[i, :]).T for i in range(batch_size)]
    )
    multimodal_embeddings = torch.cat(
        [features.unsqueeze(1).expand(-1, top_k, -1), topk_class_embeds], dim=-1
    ).to(torch.float32)
    logits = itm_head(multimodal_embeddings)
    indices_max_class_logits = torch.argmax(logits[:, :, 0], dim=-1)
    pred_classes = torch.gather(top_indices, dim=1, index=indices_max_class_logits.unsqueeze(-1))
    return pred_classes.squeeze(-1).to(torch.int64)

def _evaluate_multi_zero_shot(
    forward_func,
    classifiers,
    dataloader,
    itm_head: torch.nn.Module = None,
    top_k_preds: int = 20,
    top_k: Tuple[int, ...] = (1, 2, 5, 10),
    average: str = "micro",
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float32,
    confusion_matrix: bool = False,
    multi_label: bool = False,
    verbose: bool = False,
    template_counts=None,
    only_itm: bool = False,  # <--- NEW PARAMETER
) -> Dict:
    metrics = []
    metrics_itm = []
    for classifier in classifiers:
        num_classes = classifier.shape[1]
        metric_kwargs = dict(dist_sync_on_step=False, sync_on_compute=False)
        if multi_label:
            metric = {
                f'mAP': MultilabelAveragePrecision(num_labels=num_classes, average='macro', **metric_kwargs),
            }
        else:
            metric = {
                f'Top{k}Accuracy': MulticlassAccuracy(top_k=k, average=average, num_classes=num_classes, **metric_kwargs)
                for k in top_k
            }
        if confusion_matrix:
            metric['ConfusionMatrix'] = MulticlassConfusionMatrix(num_classes=num_classes, normalize=None, **metric_kwargs)
        if not only_itm:
            metrics.append(MetricCollection(metric).to(device))
        metrics_itm.append(MetricCollection({k: v.clone() for k, v in metric.items()}).to(device) if itm_head else None)

    classifiers = [c.to(dtype=dtype) for c in classifiers]
    start_time = time.time()
    with torch.no_grad():
        bar = tqdm(dataloader, desc=f'Predicting...', total=len(dataloader)) if verbose else dataloader
        for point in bar:
            inputs, target = point
            inputs = inputs.to(device)
            target = target.to(device)
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                features = forward_func(inputs)
            features /= features.norm(dim=-1, keepdim=True)
            for idx, classifier in enumerate(classifiers):
                if not only_itm:
                    logits_regular = features @ classifier
                    step_metric = metrics[idx](logits_regular, target.squeeze().long())
                if itm_head:
                    logits_regular = features @ classifier  # needed for ITM head
                    logits_itm = _get_itm_head_predictions(itm_head, logits_regular.clone(), top_k_preds, features, classifier)
                    step_metric_itm = metrics_itm[idx](logits_itm, target.squeeze().long())
                    # preds_regular_sorted = torch.argsort(logits_regular, dim=-1, descending=True)
                    # preds_itm_sorted = torch.argsort(logits_itm, dim=-1, descending=True)
                    # print(f"cls_{idx} - preds_regular:", preds_regular_sorted[:4, :5])
                    # print(f"cls_{idx} - preds_itm:", preds_itm_sorted[:4, :5])
                    # print(f"cls_{idx}:", step_metric["Top1Accuracy"])
                    # print(f"cls_{idx}_itm:", step_metric_itm["Top1Accuracy"])
                if verbose and average is not None:
                    if not only_itm:
                        postfix = {f"{idx}_{k}": v.item() for k, v in step_metric.items() if k != 'ConfusionMatrix'}
                        if itm_head:
                            postfix.update({f"{idx}_itm_{k}": v.item() for k, v in step_metric_itm.items() if k != 'ConfusionMatrix'})
                    else:
                        postfix = {f"{idx}_itm_{k}": v.item() for k, v in step_metric_itm.items() if k != 'ConfusionMatrix'}
                    bar.set_postfix(postfix)
    end_time = time.time()
    print(f"Predictions took {end_time - start_time:.4f} seconds")

    results = []
    for idx in range(len(classifiers)):
        result = {}
        if not only_itm:
            result["regular"] = metrics[idx].compute()
        if itm_head:
            result["itm"] = metrics_itm[idx].compute()
        if verbose:
            print(f"ZS result for classifier {idx} ({template_counts[idx]} templates): {result}")
        for key in result:
            for k, v in result[key].items():
                result[key][k] = v.cpu().numpy().item() if v.dim() == 0 else v.cpu().numpy()
        results.append(result)
    return results


if __name__ == '__main__':
    
    config_str = """
    
    model:
      image_encoder_name : 'google/vit-base-patch16-224'
      text_encoder_name : 'google-bert/bert-base-uncased'
      tokenizer :
        use_fast: False

    dataloader:
      cifar10_val:
        batch_size: 256
        shuffle: False
        num_workers: 2
        #persistent_workers: True
        pin_memory: True
    
    """
    
    device = torch.device('cuda:0')
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    
    from transformers import (
        VisionTextDualEncoderModel,
        VisionTextDualEncoderProcessor,
        AutoImageProcessor,
        AutoTokenizer,
        BertConfig,
        ViTConfig,
        VisionTextDualEncoderConfig,
        optimization
    )

    from my_datasets import Cifar10Dataset
    
    config = OmegaConf.create(config_str)
    
    model = VisionTextDualEncoderModel(
        config=VisionTextDualEncoderConfig.from_vision_text_configs(
        vision_config=ViTConfig(), 
        text_config=BertConfig()
    ))
    model.eval().to(device)

    processor = VisionTextDualEncoderProcessor(
        image_processor=AutoImageProcessor.from_pretrained(config.model.image_encoder_name), 
        tokenizer=AutoTokenizer.from_pretrained(config.model.text_encoder_name, **config.model.tokenizer)
    )
   
   
    cifar10_val = Cifar10Dataset(processor=processor)
    cifar10_dataloader = DataLoader(cifar10_val, **config.dataloader.cifar10_val)
    
    cifar10_classifier = _create_zero_shot_classifier(
            # forward_func=self.model.get_text_features,
            forward_func=lambda x: model.get_text_features(input_ids=x),
            classnames=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
            templates="a photo of a {}",
            tokenizer=processor
        )
    
    result = _evaluate_zero_shot(
            forward_func=lambda x: model.get_image_features(pixel_values=x),
            classifier=cifar10_classifier,
            dataloader=cifar10_dataloader,
            confusion_matrix=True,
            top_k=(1,)
        )
    
    print (*result.keys(),sep='\n')
    
    plt.figure(figsize=(5,5))
    plt.imshow(result['ConfusionMatrix'])
    plt.colorbar()
    plt.savefig('test1.png')
