#%%

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

# from callbacks.zeroshot_callback import _create_zero_shot_classifier,_evaluate_zero_shot
from utils.callbacks import LinearProbeCallback, ZeroShotCallback
from utils.callbacks.utils import instantiate_zeroshot_callbacks
from data.data_module import MyDataModule
from model.model_module import LitMML
from utils.utils import EmptyDataset, LightningModelWrapper

# --------------------------------- Setup ------------------------------------
def instantiate_linear_probe_callbacks(config, dataloaders):
    callbacks = dict()

    if 'cifar10' in config.datasets:
        callbacks['cifar10'] = LinearProbeCallback(
            train_dataloader=dataloaders["cifar10_train"],
            test_dataloader=dataloaders["cifar10_test"],
            linear_probe=torch.nn.Linear(512, 10),
            **config.cifar10,
        )
    if 'caltech101' in config.datasets:
        callbacks['caltech101'] = LinearProbeCallback(
            train_dataloader=dataloaders["caltech101_train"],
            test_dataloader=dataloaders["caltech101_test"],
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


 
if __name__ == '__main__':

    config = OmegaConf.load('configs/zeroshot_linear_probe_analysis_config.yaml')
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
        # linear_probe_callbacks = instantiate_linear_probe_callbacks(config.linear_probe, dataloaders)
        
            
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

    
        # linear_dict = {f"linear_{k}":v for k,v in linear_probe_callbacks.items()}
        callbacks = {f"zeroshot_{k}":v for k,v in zeroshot_callbacks.items()}
        # callbacks.update(linear_dict)
        # callbacks = zeroshot_callbacks.values() + linear_probe_callbacks.values()
        # callbacks = list(zeroshot_callbacks.values()) + list(linear_probe_callbacks.values())
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
            # if isinstance(eachcallback,LinearProbeCallback):
            #     # eachcallback.run_now_flag = True
            #     eachcallback.on_validation_epoch_start(trainer,lit_model)
            #     eachcallback.on_validation_epoch_end(trainer,lit_model)
            #     datadict[callbackname] = eachcallback.result
        return datadict
    
    data_dict = {}
    for model in model_list:
        print ('--')
        print (model)
        print ('--')
        data_dict[model_list[model]['name']] = get_model_data(model)
        
    with open('zershot_results_imagenet.p','wb') as f:
        pickle.dump(data_dict,f)


