from typing import Tuple, Union, List, Dict

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

    with torch.no_grad():
        zeroshot_weights = []
        bar = tqdm(batch_class_names, desc="Classifier weights...") if verbose else batch_class_names
        for batch_class_name in bar:
            texts = [template.format(classname) for classname in batch_class_name for template in templates]

            if tokenizer is not None:
                #texts = tokenizer(texts).to(device)  # tokenize Shape: batch_size * num_tokens x context_length
                texts = tokenizer(text=texts, padding=True, truncation=True, return_tensors="pt")['input_ids'].to(device).squeeze_()  # tokenize Shape: batch_size * num_tokens x context_length

            class_embeddings = forward_func(texts)  # batch_size * num_tokens x embedding_dim
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


def _evaluate_zero_shot(forward_func,
                        classifier,
                        dataloader,
                        top_k: Tuple[int, ...] = (1, 2, 5, 10),
                        average: str = "micro",
                        device: Union[str, torch.device] = "cuda",
                        dtype: torch.dtype = torch.float32,
                        confusion_matrix: bool = False,
                        multi_label: bool = False,
                        verbose: bool = False) -> Dict:
    num_classes = classifier.shape[1]  # classifier shape (embed, num_classes)

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

    classifier = classifier.to(dtype=dtype)
    with torch.no_grad():
        bar = tqdm(dataloader, desc=f'Predicting...', total=len(dataloader)) if verbose else dataloader
        for point in bar:
            inputs, target = point
            inputs = inputs.to(device)
            target = target.to(device)

            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                features = forward_func(inputs)

            features /= features.norm(dim=-1, keepdim=True)
            logits = features @ classifier

            step_metric = metric(logits, target.squeeze().long())

            if verbose and average is not None:
                bar.set_postfix({k: v.item() for k, v in step_metric.items() if k != 'ConfusionMatrix'})

    result = metric.compute()
    if verbose:
        print(f"ZS result {result}")

    for k, v in result.items():
        result[k] = v.cpu().numpy().item() if v.dim() == 0 else v.cpu().numpy()
    return result


def make_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]





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
