import rootutils
rootutils.setup_root('.', indicator=".project-root", pythonpath=True)
from src.model.model_module import LitMML, MLMWrapper
from src.model.utils import get_model_and_processor
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from src.data.data_module import MyDataModule
from src.callbacks.zeroshot_callback import (
    _create_zero_shot_classifier, 
    _evaluate_zero_shot,
    _get_itm_head_predictions
)
from torchmetrics.classification import (
    MulticlassAccuracy,
    # MultilabelAveragePrecision,
    # MulticlassConfusionMatrix,
)
from torchmetrics import MetricCollection


device = "cuda:3"
# ckpt_path = '/home/data/bhavin/ckpts/93t3xgrr/last.ckpt' # CLIP+ITM
ckpt_path = "/home/data/bhavin/higher_augmentations_ckpts/7m9tx2jf/last.ckpt" # CLIP+ITM-aug
model, processor = get_model_and_processor(config=OmegaConf.load('configs/model/dual_encoder.yaml'))
model = MLMWrapper(model)
lit_model = LitMML.load_from_checkpoint(ckpt_path, model=model, processor=processor, map_location=device)
model, processor = lit_model.model, lit_model.processor
itm_head = lit_model.itm_head

data_config = OmegaConf.load('configs/data/test.yaml')
data_config.root = '/home/data'
data_config.dataloader.test.batch_size = 128 #16
data_config.datasets = ["Caltech101"]
data_module = MyDataModule(data_config=data_config, processor=processor)
dataloaders = data_module.get_test_dataloaders()

loader = dataloaders["Caltech101"]["test"]
class_names = loader.dataset.classnames
num_classes = len(class_names)
# templates = ["a photo of a {}."]
templates = [
    'a photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.'
]
verbose = True
dtype = torch.float32
forward_func = lambda x: model.get_image_features(pixel_values=x)
top_k_preds = 10

classifier = _create_zero_shot_classifier(
    forward_func=lambda x, y: model.get_text_features(input_ids=x, attention_mask=y),
    classnames=class_names,
    templates=templates,
    tokenizer=processor,
    batch_size=loader.batch_size,
    device=device,
    verbose=verbose,
)


top_k = [1, 3, 5]
metric_kwargs = dict(dist_sync_on_step=False, sync_on_compute=False)
metric = {
            f'Top{k}Accuracy': MulticlassAccuracy(top_k=k, average="micro", num_classes=num_classes, **metric_kwargs)
            for k in top_k
        }
metric = MetricCollection(metric).to(device)
# metric_itm = MetricCollection(metric.clone()).to(device)
metric_itm = MetricCollection({k: v.clone() for k, v in metric.items()}).to(device)

with torch.no_grad():
    bar = tqdm(loader, desc=f'Predicting...', total=len(loader)) if verbose else loader
    for point in bar:
        inputs, target = point
        inputs = inputs.to(device)
        target = target.to(device)
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            features = forward_func(inputs)
        features /= features.norm(dim=-1, keepdim=True)
        logits_regular = features @ classifier
        step_metric = metric(logits_regular, target.squeeze().long())
        # Get sorted class indices for logits_regular
        preds_regular_sorted = torch.argsort(logits_regular, dim=-1, descending=True)
        if itm_head:
            logits_itm = _get_itm_head_predictions(itm_head, logits_regular.clone(), top_k_preds, features, classifier)
            step_metric_itm = metric_itm(logits_itm, target.squeeze().long())
            # Get sorted class indices for logits_itm
            preds_itm_sorted = torch.argsort(logits_itm, dim=-1, descending=True)
        if verbose:
            postfix = {f"{k}": v.item() for k, v in step_metric.items() if k != 'ConfusionMatrix'}
            if itm_head:
                postfix.update({f"itm_{k}": v.item() for k, v in step_metric_itm.items() if k != 'ConfusionMatrix'})
            bar.set_postfix(postfix)
        # Print or log the sorted indices for the first sample in the batch
        print("Sample 0 - contrastive sorted indices:", preds_regular_sorted[0].cpu().tolist())
        if itm_head:
            print("Sample 0 - ITM head sorted indices:", preds_itm_sorted[0].cpu().tolist())
