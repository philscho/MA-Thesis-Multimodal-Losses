# %%
import json
from pathlib import Path
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
)
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.strategies import DDPStrategy
import os
import torch
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    ViTConfig,
    BertConfig,
    VisionTextDualEncoderConfig,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
)
from omegaconf import OmegaConf
import pickle

from tqdm import tqdm
from torchvision.datasets import ImageNet

from model_module import LitMML

from torchvision.datasets import (
    SBU,
    Caltech101,
    Caltech256,
    CIFAR100,
    CIFAR10,
    Cityscapes,
    OxfordIIITPet,
    StanfordCars,
    STL10,
    Flowers102,
    Food101,
    FGVCAircraft,
    DTD,
)

from torch.utils.data import DataLoader

from utils.callbacks.zeroshot_callback import (
    _create_zero_shot_classifier,
    _evaluate_zero_shot,
)
from utils.callbacks import LinearProbeCallback, ZeroShotCallback

# from callbacks.utils import instantiate_zeroshot_callbacks
# from data_module import MyDataModule
from model_module import LitMML

# from utils.utils import EmptyDataset, LightningModelWrapper

# # --------------------------------- Setup ------------------------------------
# def instantiate_linear_probe_callbacks(config, dataloaders):
#     callbacks = dict()

#     if 'cifar-10' in config.datasets:
#         callbacks['cifar-10'] = LinearProbeCallback(
#             train_dataloader=dataloaders["cifar-10_train"],
#             test_dataloader=dataloaders["cifar-10_test"],
#             linear_probe=torch.nn.Linear(512, 10),
#             **config.cifar10,
#         )
#     if 'caltech-101' in config.datasets:
#         callbacks['caltech-101'] = LinearProbeCallback(
#             train_dataloader=dataloaders["caltech-101_train"],
#             test_dataloader=dataloaders["caltech-101_test"],
#             linear_probe=torch.nn.Linear(512, 101),
#             **config.caltech101,
#         )

#     return callbacks


def get_models(config, ckpt_path):
    model = VisionTextDualEncoderModel(
        config=VisionTextDualEncoderConfig.from_vision_text_configs(
            vision_config=ViTConfig(), text_config=BertConfig()
        )
    )

    # model = VisionTextDualEncoderModel.from_vision_text_pretrained(config.model.image_encoder_name, config.model.text_encoder_name)
    processor = VisionTextDualEncoderProcessor(
        image_processor=AutoImageProcessor.from_pretrained(
            config.model.image_encoder_name, input_data_format="channels_last"
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            config.model.text_encoder_name, **config.model.tokenizer
        ),
    )

    lit_model = LitMML.load_from_checkpoint(
        ckpt_path,
        model=model,
        processor=processor,
        # loss_cfg=config.loss,
        # optimizer_cfg=config.optimizer,
        # scheduler_cfg=config.scheduler,
        strict=True,
    )
    return lit_model


DATASET_ROOT = Path("/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/")

all_datasets = {
    ### 'SBU' : SBU(root=DATASET_ROOT/'SBU-Captions'),
    ### 'CityScapes':Cityscapes(root=DATASET_ROOT/'Cityscapes',split='test'),
    ### "Flowers102": Flowers102(root=DATASET_ROOT, split="test"),  #No labels, cannot do zeroshot!
    # "Caltech256": Caltech256(root=DATASET_ROOT), # needs some fixing
    "Caltech101": Caltech101(root=DATASET_ROOT),
    "CIFAR10": CIFAR10(root=DATASET_ROOT, train=False),
    "CIFAR100": CIFAR100(root=DATASET_ROOT, train=False),
    "DTD": DTD(root=DATASET_ROOT, split="test"),
    "OxfordsIIITPet": OxfordIIITPet(root=DATASET_ROOT, split="test", download=True),
    "StanfordCars": StanfordCars(root=DATASET_ROOT, split="test"),
    "FGVCAircraft": FGVCAircraft(root=DATASET_ROOT, split="test"),
    "Food101": Food101(root=DATASET_ROOT, split="test"),
    "STL10": STL10(root=DATASET_ROOT, split="test"),
    # 'ImageNet' : ImageNet(root=DATASET_ROOT/'ImageNet',split='val'),   
}




class TorchVisionDatasetWrapper:
    def __init__(self, baseclass, processor=None, transform=None):
        self.baseclass = baseclass
        self.processor = processor
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.baseclass.__getitem__(idx)
        image = image.convert("RGB")

        if self.processor is not None:
            image = torch.squeeze(
                self.processor(images=image, return_tensors="pt")["pixel_values"]
            )
            return image, label
        else:
            if self.transform:
                image = self.transform(image)
            return image, label

    def __len__(self):
        return self.baseclass.__len__()



if __name__ == '__main__':

    config_str = """
    model:
        image_encoder_name: 'google/vit-base-patch16-224'
        text_encoder_name: 'google-bert/bert-base-uncased'
        tokenizer:
            use_fast: False
            
    lightning:
        seed: 69
        trainer:
            fast_dev_run: True #True #False #True #False
            log_every_n_steps: 1 #5 #10
            max_epochs: 1
            devices: 2  
            num_nodes: 1
            accelerator: 'gpu'
            strategy: 'ddp' #'ddp_find_unused_parameters_true'
            deterministic: 'warn' #True
            precision: '16-mixed'
            gradient_clip_algorithm: "norm"
            gradient_clip_val: 1.0   

    """

    config = OmegaConf.create(config_str)
    print (config)
    
    model_list = OmegaConf.load('/pfss/mlde/workspaces/mlde_wsp_PI_Roig/bhavin/students/phillipscholl/multimodal/analysis/model_info.yaml')
    
    def get_model_data(model_id):
        ckpt_path = f"/pfss/mlde/workspaces/mlde_wsp_PI_Roig/bhavin/students/phillipscholl/multimodal/models/cliplike/ckpts/{model_id}/last.ckpt"
        lit_model = get_models(config, ckpt_path)
        model, processor = lit_model.model, lit_model.processor
            
        callbacks = {}
        for datasetname, dataset in all_datasets.items():
            wrapped_dataset = TorchVisionDatasetWrapper(
                dataset, processor=processor, transform=None
            )
            dataloader = DataLoader(wrapped_dataset, batch_size=64, shuffle=False)
            # images, text = next(iter(dataloader))
            # print(datasetname, images.shape)

            # if datasetname == 'CIFAR10':
                # classnames = dataset.

            if datasetname =='Caltech256':
                classnames = [x.split('.')[1] for x in dataset.categories]
            elif datasetname =='Caltech101':
                classnames = dataset.categories
            elif datasetname == 'ImageNet':
                classnames = json.load(open('/pfss/mlde/workspaces/mlde_wsp_PI_Roig/bhavin/students/phillipscholl/multimodal/my_datasets/imagenet-simple-labels.json'))
            elif datasetname == 'Flowers102':
                classnames = []
                raise NotImplementedError
            else:
                classnames = dataset.classes



            callbacks[datasetname] = ZeroShotCallback(
                dataloader=dataloader,
                tokenizer=processor,
                text_forward=lambda x, y: model.get_text_features(
                    input_ids=x, attention_mask=y
                ),
                # text_forward=lambda x: model.get_text_features(input_ids=x),
                modality_forward=lambda x: model.get_image_features(pixel_values=x),
                dataset_name=datasetname,
                classnames=classnames,
                templates=None,
                batch_size=64,
                device=torch.device("cuda:0"),
                top_k=[1, 3, 5],
                confusion_matrix=False,
                verbose=True,
            )

            

        trainer = pl.Trainer(
            callbacks=list(callbacks.values()),
            inference_mode=False,  # to allow the training of the linear probe
            **config.lightning.trainer,
        )
        
        datadict = {}
        for callbackname, eachcallback in callbacks.items():
            print ()
            print (callbackname)
            if isinstance(eachcallback,ZeroShotCallback):
                eachcallback.on_validation_end(trainer,lit_model)
                datadict[eachcallback.dataset_name] = eachcallback.result
            # if isinstance(eachcallback,LinearProbeCallback):
            #     eachcallback.on_validation_epoch_start(trainer,lit_model)
            #     eachcallback.on_validation_epoch_end(trainer,lit_model)
            #     datadict[callbackname] = eachcallback.result


        with open(f"{model_id}_intermediate_all_datasets.p",'wb') as f:
            pickle.dump(datadict,f)

        return datadict


    data_dict = {}
    for model in tqdm(list(model_list.keys())[::-1]):
        print ('--')
        print (model)
        print ('--')
        data_dict[model_list[model]['name']] = get_model_data(model)
        
    with open('all_models_all_datasets_expanded_datasets_file.p','wb') as f:
        pickle.dump(data_dict,f)


# %%
