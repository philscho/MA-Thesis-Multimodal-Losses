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
from callbacks.linear_probe_callback import create_linear_probe,eval_linear_probe 
from callbacks.utils import instantiate_zeroshot_callbacks
from data_module import MyDataModule
from model_module import LitMML
from utils.utils import EmptyDataset, LightningModelWrapper

# --------------------------------- Setup ------------------------------------

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



config = OmegaConf.load('configs/zeroshot_config_g4_tests.yaml')
zeroshot_config = OmegaConf.load('configs/zeroshot.yaml')

config = OmegaConf.merge(config,zeroshot_config)
OmegaConf.resolve(config)

model_list = OmegaConf.load('/pfss/mlde/workspaces/mlde_wsp_PI_Roig/bhavin/students/phillipscholl/multimodal/analysis/model_info.yaml')




def get_model_specific_information(modelname):

    model, processor = get_models(config)
    data_module = MyDataModule(config, 
                            processor, 
                            augmentation=transforms.RandAugment(),
                            num_views=2,
                            )
    dataloaders = data_module.callback_dataloader()


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


        device = torch.device('cuda:0')
        model.to(device)
        model.eval()

        train_dataloader=dataloaders["caltech-101_train"]
        test_dataloader=dataloaders["caltech-101_test"]
        linear_probe=torch.nn.Linear(768, 101)

        trained_linear_probe = create_linear_probe(
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
                            num_classes=101,
                            confusion_matrix=False,
                            top_k=(1,3,5),
                            verbose=True,
                            device=device,
                        )

        print (result)
        layer_data_dict[MAIN_LAYER_KEY] = result
        
    return layer_data_dict

data_dict = {}
for model in model_list:
    print ('--')
    print (model)
    print ('--')
    data_dict[model_list[model]['name']] = get_model_specific_information(model)


with open('caltech101_layespecific_linear_results.p','wb') as f:
    pickle.dump(data_dict,f)


#%%