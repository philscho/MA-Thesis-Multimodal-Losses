# %%

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
    VisionTextDualEncoderProcessor
)
from omegaconf import OmegaConf
import pickle


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

# from callbacks.zeroshot_callback import _create_zero_shot_classifier,_evaluate_zero_shot
from utils.callbacks import LinearProbeCallback, ZeroShotCallback
from utils.callbacks.utils import instantiate_zeroshot_callbacks
from data_module import MyDataModule
from model_module import LitMML
from utils.utils import EmptyDataset, LightningModelWrapper

# --------------------------------- Setup ------------------------------------
def instantiate_linear_probe_callbacks(config, dataloaders):
    callbacks = dict()

    if 'cifar-10' in config.datasets:
        callbacks['cifar-10'] = LinearProbeCallback(
            train_dataloader=dataloaders["cifar-10_train"],
            test_dataloader=dataloaders["cifar-10_test"],
            linear_probe=torch.nn.Linear(512, 10),
            **config.cifar10,
        )
    if 'caltech-101' in config.datasets:
        callbacks['caltech-101'] = LinearProbeCallback(
            train_dataloader=dataloaders["caltech-101_train"],
            test_dataloader=dataloaders["caltech-101_test"],
            linear_probe=torch.nn.Linear(512, 101),
            **config.caltech101,
        )
    
    
    
    
    
    return callbacks

def get_models(config):
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
    return model, processor


DATASET_ROOT = Path('/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/')

all_datasets = {
    'SBU' : SBU(root=DATASET_ROOT/'SBU-Captions'),
    'Caltech101':Caltech101(root=DATASET_ROOT/'caltech101'),
    'Caltech256':Caltech256(root=DATASET_ROOT/'caltech256',download=True),
    'CIFAR10': CIFAR10(root=DATASET_ROOT/'cifar10',train=False),
    'CIFAR100': CIFAR100(root=DATASET_ROOT/'cifar100',train=False),
    'DTD': DTD(root=DATASET_ROOT/'dtd',split='test'),
    'CityScapes':Cityscapes(root=DATASET_ROOT/'Citescapes',split='test'),
    'OxfordsIIITPet':OxfordIIITPet(root=DATASET_ROOT/'oxford-iiit-pet',split='test'),
    'StanfordCars':StanfordCars(root=DATASET_ROOT/'stanford_cars',split='test'),
    'Flowers102': Flowers102(root=DATASET_ROOT/'flowers-102',split='test'),
    'FGVCAircraft':FGVCAircraft(root=DATASET_ROOT/'fgvc-aircraft-2013b',split='test'),
    'Food101':Food101(root=DATASET_ROOT/'food-101',split='test'),
    'STL10':STL10(root=DATASET_ROOT/'stl10_binary',split='test'),
}




if __name__ == '__main__':

    config = OmegaConf.load('configs/zeroshot_linear_probe_analysis_config.yaml')
    # zeroshot_config = OmegaConf.load('configs/zeroshot.yaml')

    # config = OmegaConf.merge(config)
    OmegaConf.resolve(config)

    model_list = OmegaConf.load('/pfss/mlde/workspaces/mlde_wsp_PI_Roig/bhavin/students/phillipscholl/multimodal/analysis/model_info.yaml')


    def get_model_data(modelname):
        model, processor = get_models(config)
        data_module = MyDataModule(config, 
                                processor, 
                                augmentation=transforms.RandAugment(),
                                num_views=2,
                                )
        dataloaders = data_module.callback_dataloader()
        zeroshot_callbacks = instantiate_zeroshot_callbacks(config.zeroshot, dataloaders, model, processor)
        linear_probe_callbacks = instantiate_linear_probe_callbacks(config.linear_probe, dataloaders)
        
            
        ckpt_path = f"/pfss/mlde/workspaces/mlde_wsp_PI_Roig/bhavin/students/phillipscholl/multimodal/models/cliplike/ckpts/{modelname}/last.ckpt"
        lit_model = LitMML.load_from_checkpoint(ckpt_path, 
                                                model=model, 
                                                processor=processor,
                                                # loss_cfg=config.loss,
                                                # optimizer_cfg=config.optimizer,
                                                # scheduler_cfg=config.scheduler,
                                                # strict=False,
                                                )
        model, processor = lit_model.model, lit_model.processor


    
        linear_dict = {f"linear_{k}":v for k,v in linear_probe_callbacks.items()}
        callbacks = {f"zeroshot_{k}":v for k,v in zeroshot_callbacks.items()}
        callbacks.update(linear_dict)
        print (callbacks)


        trainer = pl.Trainer(
            callbacks=list(callbacks.values()),
            inference_mode=False,  # to allow the training of the linear probe
            **config.lightning.trainer,
        )
        datadict = {}
        for callbackname, eachcallback in callbacks.items():
            if isinstance(eachcallback,ZeroShotCallback):
                eachcallback.on_validation_end(trainer,lit_model)
                datadict[eachcallback.dataset_name] = eachcallback.result
            if isinstance(eachcallback,LinearProbeCallback):
                eachcallback.on_validation_epoch_start(trainer,lit_model)
                eachcallback.on_validation_epoch_end(trainer,lit_model)
                datadict[callbackname] = eachcallback.result
        return datadict
    
    data_dict = {}
    for model in model_list[::-1]:
        print ('--')
        print (model)
        print ('--')
        data_dict[model_list[model]['name']] = get_model_data(model)
        
    with open('linear_results.p','wb') as f:
        pickle.dump(data_dict,f)
 # needs some fixingc