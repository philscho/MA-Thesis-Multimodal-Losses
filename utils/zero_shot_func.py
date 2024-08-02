from typing import List, Tuple, Dict, Union

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MultilabelAveragePrecision, MulticlassConfusionMatrix
from tqdm import tqdm


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
