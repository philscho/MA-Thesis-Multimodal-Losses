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
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf
import pickle
import wandb


# from callbacks.zeroshot_callback import _create_zero_shot_classifier,_evaluate_zero_shot
from src.callbacks import LinearProbeCallback, ZeroShotCallback
from src.callbacks.linear_probe_callback import train_linear_probe, eval_linear_probe
from src.callbacks.utils import instantiate_zeroshot_callbacks, instantiate_linear_probe_callbacks, instantiate_zeroshot_callbacks_for_all_text_layers
from src.data.data_module import MyDataModule
from src.data.utils import EmptyDataset
from src.model.model_module import LitMML, MLMWrapper
from src.model.utils import get_model_and_processor
from src.utils.utils import LightningModelWrapper

def get_model_data(ckpt_path,
                   model_config, 
                   data_config, 
                   callbacks_config,
                   trainer_config,
                   logger_config,
                   ckpt_path_2=None,
                   loss_config=None,
                   optimizer_config=None,
                   scheduler_config=None,
                   ):
    
    device_idx = trainer_config.devices[0]
    
    model, processor = get_model_and_processor(model_config)
    # if "0.1_and_0.5_ckpts" in ckpt_path or "higher_augmentations_ckpts" in ckpt_path:
    #     model = MLMWrapper(model)
    model = MLMWrapper(model)
    lit_model = LitMML.load_from_checkpoint(ckpt_path, 
                                            model=model, 
                                            processor=processor,
                                            map_location=f"cuda:{device_idx}",
                                            )
    model, processor = lit_model.model, lit_model.processor

    if ckpt_path_2 is not None:
        print(f"Loading text model from {ckpt_path_2}")
        text_ckpt = torch.load(ckpt_path_2, map_location='cpu')
        state_dict = text_ckpt['state_dict']
        for key in list(state_dict.keys()):
            if not key.startswith("model.basemodel.text"):
                state_dict.pop(key)
        model.basemodel.text_projection.load_state_dict({"weight": state_dict.pop("model.basemodel.text_projection.weight")})
        model.basemodel.text_model.load_state_dict({k.replace("model.basemodel.text_model.", ""): v for k, v in state_dict.items()})
    
    if "zeroshot" in callbacks_config:
        if callbacks_config.zeroshot is not None:
            if callbacks_config.zeroshot.use_itm_head:
                itm_head = getattr(lit_model, 'itm_head', None)
            else: itm_head = None
    else: itm_head = None
    
    data_module = MyDataModule(data_config,
                               processor,
                               augmentation=transforms.RandAugment(),
                               num_views=2,
                               )
    dataloaders = data_module.get_test_dataloaders()
    
    callbacks = dict()
    # if "zeroshot" in callbacks_config:
    if callbacks_config.zeroshot:
        if callbacks_config.zeroshot.eval_all_text_layers:
            zeroshot_callbacks = instantiate_zeroshot_callbacks_for_all_text_layers(
                zeroshot_config=callbacks_config.zeroshot.callback,  
                dataloaders=dataloaders, 
                model=model,
                processor=processor, 
                itm_head=itm_head,
            )
        else:
            zeroshot_callbacks = instantiate_zeroshot_callbacks(
                zeroshot_config=callbacks_config.zeroshot.callback, 
                dataloaders=dataloaders, 
                model=model, 
                processor=processor, 
                itm_head=itm_head,
            )
        callbacks.update({f"zeroshot_{k}":v for k,v in zeroshot_callbacks.items()})
    # if "linear_probe" in callbacks_config:
    if callbacks_config.get("linear_probe", None):
        linear_probe_callbacks = instantiate_linear_probe_callbacks(callbacks_config.linear_probe, dataloaders)
        callbacks.update({f"linear_{k}":v for k,v in linear_probe_callbacks.items()})
    
    logger = WandbLogger(**logger_config)

    trainer = pl.Trainer(
        #devices=3,
        logger=logger,
        callbacks=list(callbacks.values()),
        inference_mode=False,  # to allow the training of the linear probe
        **trainer_config,
    )

    trainer.validate(lit_model, datamodule=data_module)

    datadict = {}
    for name, callback in callbacks.items():
        if isinstance(callback, ZeroShotCallback):
            # Store the structured logged_results for later saving
            datadict[name] = getattr(callback, "logged_results", [])
        if isinstance(callback, LinearProbeCallback):
            datadict[name] = callback.result
    
    wandb.finish()

    return datadict



def eval_linear_probe_per_layer(config, ckpt):
    model, processor = get_model_and_processor(config)
    ckpt_path = os.path.join(config.paths.ckpts_dir, ckpt, "last.ckpt")
    lit_model = LitMML.load_from_checkpoint(ckpt_path, 
                                            model=model, 
                                            processor=processor
                                            )
    model, processor = lit_model.model, lit_model.processor

    data_module = MyDataModule(config,
                                processor, 
                                augmentation=transforms.RandAugment(),
                                num_views=2,
                                )
    dataloaders = data_module.get_test_dataloaders()
    train_dataloader = dataloaders["CIFAR10"]["train"]
    test_dataloader = dataloaders["CIFAR10"]["test"]
    num_classes = train_dataloader.dataset.num_classes

    INTERESTING_LAYERS = {
        "encoder_layer_0": model.vision_model.encoder.layer[0],
        "encoder_layer_1": model.vision_model.encoder.layer[1],
        "encoder_layer_2": model.vision_model.encoder.layer[2],
        "encoder_layer_3": model.vision_model.encoder.layer[3],
        "encoder_layer_4": model.vision_model.encoder.layer[4],
        "encoder_layer_5": model.vision_model.encoder.layer[5],
        "encoder_layer_6": model.vision_model.encoder.layer[6],
        "encoder_layer_7": model.vision_model.encoder.layer[7],
        "encoder_layer_8": model.vision_model.encoder.layer[8],
        "encoder_layer_9": model.vision_model.encoder.layer[9],
        "encoder_layer_10": model.vision_model.encoder.layer[10],
        "encoder_layer_11": model.vision_model.encoder.layer[11],
        # "fc_norm": model.fc_norm,
    }

    device = torch.device('cuda:{0}'.format(config.lightning.trainer.devices[0]))

    layer_data_dict = {}
    for MAIN_LAYER_KEY in INTERESTING_LAYERS:
        def get_reps(x) -> dict[torch.Tensor]:
            # attach hooks to the intermediate layers of the encoder
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output[0].detach()
                return hook

            for k, v in INTERESTING_LAYERS.items():
                if k == MAIN_LAYER_KEY:
                    v.register_forward_hook(get_activation(k))

            activation = {}
            output = model.get_image_features(pixel_values=x)
            # assert (
                # len(activation.keys()) == 13
            # ), f"wrong number of keys found : {activation.keys()}"
            # activation["output"] = output
            return activation

        def custom_forward_func(x):
            outs = get_reps(x)[MAIN_LAYER_KEY]
            return outs[:,0,:] # return class token

        model.to(device)
        model.eval()

        linear_probe=torch.nn.Linear(768, num_classes)

        trained_linear_probe = train_linear_probe(
                        forward_func=custom_forward_func,
                        dataloader=train_dataloader,
                        linear_layer=linear_probe,
                        max_epochs=500,
                        verbose=True,
                        device=device,
                    )

        result = eval_linear_probe(
                            forward_func=custom_forward_func,
                            classifier=trained_linear_probe,
                            dataloader=test_dataloader,
                            num_classes=num_classes,
                            confusion_matrix=False,
                            top_k=(1,3,5),
                            verbose=True,
                            device=device,
                        )

        print(result)
        layer_data_dict[MAIN_LAYER_KEY] = result

    return layer_data_dict
