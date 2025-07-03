#%%
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
import os
import torch
from torchvision import transforms
import hydra
from omegaconf import OmegaConf
import pickle
import rootutils
from typing import Dict
from tqdm import tqdm
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.callbacks.linear_probe_callback import train_linear_probe, eval_linear_probe, setup_linear_classifiers
from src.data.data_module import MyDataModule
from src.model.model_module import LitMML, MLMWrapper
from src.model.utils import get_model_and_processor


@hydra.main(version_base=None, config_path="../../configs", config_name="linear_per_layer")
def main(config):
    print(OmegaConf.to_yaml(config))

    torch.set_float32_matmul_precision("medium")
    seed_everything(config.lightning.seed, workers=True)
    
    ckpt_list = config.checkpoints
    dataset = config.dataset
    
    def eval_linear_probe_per_layer(config, ckpt_path):
        device = torch.device('cuda:{0}'.format(config.lightning.trainer.devices[0]))
        
        model, processor = get_model_and_processor(config.model)
        model = MLMWrapper(model)
        lit_model = LitMML.load_from_checkpoint(ckpt_path, 
                                                model=model, 
                                                processor=processor,
                                                map_location=device,
                                                )
        model, processor = lit_model.model, lit_model.processor
        
        data_module = MyDataModule(config.data,
                                   processor, 
                                   augmentation=transforms.RandAugment(),
                                   num_views=2,
                                   )
        dataloaders = data_module.get_test_dataloaders()
        train_dataloader = dataloaders[dataset]["train"]
        test_dataloader = dataloaders[dataset]["test"]
        num_classes = train_dataloader.dataset.num_classes

        INTERESTING_LAYERS = {
            "encoder_layer_0": model.basemodel.vision_model.encoder.layer[0],
            "encoder_layer_1": model.basemodel.vision_model.encoder.layer[1],
            "encoder_layer_2": model.basemodel.vision_model.encoder.layer[2],
            "encoder_layer_3": model.basemodel.vision_model.encoder.layer[3],
            "encoder_layer_4": model.basemodel.vision_model.encoder.layer[4],
            "encoder_layer_5": model.basemodel.vision_model.encoder.layer[5],
            "encoder_layer_6": model.basemodel.vision_model.encoder.layer[6],
            "encoder_layer_7": model.basemodel.vision_model.encoder.layer[7],
            "encoder_layer_8": model.basemodel.vision_model.encoder.layer[8],
            "encoder_layer_9": model.basemodel.vision_model.encoder.layer[9],
            "encoder_layer_10": model.basemodel.vision_model.encoder.layer[10],
            "encoder_layer_11": model.basemodel.vision_model.encoder.layer[11],
            "projection_layer": model.basemodel.visual_projection,
            # "fc_norm": model.fc_norm,
        }

        layer_data_dict = {}
        for MAIN_LAYER_KEY in tqdm(INTERESTING_LAYERS, desc="Going through the layers"):
            def get_reps(x):
                # attach hooks to the intermediate layers of the encoder
                activation = {}
                if MAIN_LAYER_KEY == "projection_layer":
                    def get_activation(name):
                        def hook(model, input, output):
                            activation[name] = output.detach()
                        return hook
                else:
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

            if MAIN_LAYER_KEY == "projection_layer":
                hidden_size = 512
                def custom_forward_func(x):
                    outs = get_reps(x)[MAIN_LAYER_KEY]
                    return outs # return class token
            else:
                hidden_size = 768
                def custom_forward_func(x):
                    outs = get_reps(x)[MAIN_LAYER_KEY]
                    return outs[:,0,:] # return class token

            model.to(device)
            model.eval()

            print(f"{MAIN_LAYER_KEY}")

            # linear_probe = torch.nn.Linear(hidden_size, num_classes)
            classifiers, params = setup_linear_classifiers(out_dim=hidden_size, num_classes=num_classes, learning_rates=[0.0001])

            trained_linear_probe = train_linear_probe(
                            forward_func=custom_forward_func,
                            dataloader=train_dataloader,
                            dataset_name=dataset,
                            classifiers=classifiers,
                            parameters=params,
                            # linear_layer=linear_probe,
                            max_epochs=500,
                            verbose=True,
                            device=device,
                        )
            
            result = eval_linear_probe(
                                forward_func=custom_forward_func,
                                classifiers=trained_linear_probe,
                                dataloader=test_dataloader,
                                num_classes=num_classes,
                                confusion_matrix=False,
                                top_k=(1,3,5),
                                verbose=True,
                                device=device,
                            )
            
            layer_data_dict[MAIN_LAYER_KEY] = result

        return layer_data_dict

    print(f"Running for checkpoints: {ckpt_list}")
    for model_id, model_info in ckpt_list.items():
        print ('--')
        print (model_id)
        print ('--')
        if "path" in model_info:
            ckpt_path = model_info["path"]
        elif model_id == "n5trwpk9":
            ckpt_path = "/home/data/bhavin/n5trwpk9/ckpt-epoch=34-loss-val=4.378.ckpt"
        elif model_id == "zathvtrx":
            ckpt_path = "/data/bhavin/ckpts_old/zathvtrx/ckpt-epoch=14-loss-val=0.002.ckpt"
        else:
            ckpt_path = os.path.join(config.paths.ckpts_dir, model_id, "last.ckpt")
        data_dict = eval_linear_probe_per_layer(config, ckpt_path)
        with open(f"{dataset}-{model_id}-linear_per_layer.p", 'wb') as f:
            pickle.dump(data_dict, f)


if __name__ == '__main__':
    main()
