import pickle
import os
import torch
import rootutils
from omegaconf import OmegaConf
from torchvision.transforms import RandAugment
import hydra

ROOT = rootutils.setup_root("/home/phisch/multimodal", indicator=".project-root", pythonpath=True)

from centered_kernel_alignment.src.ckatorch import CKA
from src.model.utils import get_model_and_processor
from src.model.model_module import LitMML, MLMWrapper
from src.data.data_module import MyDataModule


@hydra.main(version_base=None, config_path="configs", config_name="cka")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device(f"cuda:{cfg.device[0]}")

    model_config = cfg.model
    model, processor = get_model_and_processor(model_config)
    model = MLMWrapper(model)
    use_pretrained_model = cfg.get("use_pretrained_model", False)
    if use_pretrained_model:
        model_pretrained, _ = get_model_and_processor(model_config, pretrained=True)
        model_pretrained = MLMWrapper(model_pretrained)
    
    if cfg.get("model_pairs", None) is not None:
        model_pairs = cfg.model_pairs
        print(f"Using model pairs: {model_pairs}")
    else:
        model_pairs = []
        models = list(cfg.checkpoints.keys())
        for m1 in range(len(models)):
            model_pairs.append((models[m1], models[m1]))
            for m2 in range(len(models)):
                if m2 > m1:
                    model_pairs.append((models[m1], models[m2]))
        
    def get_ckpt_path(model_id):
        if cfg.checkpoints[model_id].get("path", None) is not None:
            ckpt_path = cfg.checkpoints[model_id].path
        elif model_id == "n5trwpk9":
            ckpt_path = "/home/data/bhavin/n5trwpk9/ckpt-epoch=34-loss-val=4.378.ckpt"
        elif model_id == "zathvtrx":
            ckpt_path = "/data/bhavin/ckpts_old/zathvtrx/ckpt-epoch=14-loss-val=0.002.ckpt"
        else:
            ckpt_dir = os.path.join(cfg.paths.ckpts_dir, model_id)
            ckpt_files = os.listdir(ckpt_dir)
            if len(ckpt_files) > 1 and "last.ckpt" in ckpt_files:
                ckpt_files.remove("last.ckpt")
            ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
        return ckpt_path
    
    for (model_id_1, model_id_2) in model_pairs:
        ckpt_path_m1 = get_ckpt_path(model_id_1)
        ckpt_path_m2 = get_ckpt_path(model_id_2)
    
        lit_model_1 = LitMML.load_from_checkpoint(ckpt_path_m1, 
                                                model=model, 
                                                processor=processor,
                                                )
        model1 = lit_model_1.model.basemodel
        print(f"Loaded model 1 ({model_id_1})")

        if use_pretrained_model:
            lit_model = LitMML(model=model_pretrained,
                               processor=processor,
                               loss_cfg=cfg.loss,
                               optimizer_cfg=cfg.optimizer,
                               scheduler_cfg=cfg.scheduler,
                               )
            model2 = lit_model.model.basemodel
            print(f"Loaded pretrained model 2 ({model_id_2})")
        else:
            lit_model_2 = LitMML.load_from_checkpoint(ckpt_path_m2, 
                                                    model=model, 
                                                    processor=processor,
                                                    )
            model2 = lit_model_2.model.basemodel
            print(f"Loaded model 2 ({model_id_2})")

        model1_name = cfg.checkpoints[model_id_1].model
        model2_name = cfg.checkpoints[model_id_2].model
        
        #vision_layers = [f"basemodel.vision_model.encoder.layer.{i}" for i in range(12)] + ["basemodel.vision_model.pooler", "basemodel.visual_projection"]
        vision_layers = [f"vision_model.encoder.layer.{i}" for i in range(12)]# + ["vision_model.pooler", "visual_projection"]
        text_layers = [f"text_model.encoder.layer.{i}" for i in range(12)] + ["text_model.pooler", "text_projection"]
        # print(layers)
        cka = CKA(
            first_model=model1,
            second_model=model2,
            layers=vision_layers,
            second_layers=vision_layers,
            first_name=model1_name,
            second_name=model2_name,
            device=device,
        )
        model1_forward = model1.get_image_features
        model2_forward = model2.get_image_features
        # cka.second_model = lambda x: model2.get_image_features(pixel_values=x)
        # cka.model2 = lambda x: model2.get_text_features(input_ids=x)
        
        data_config = cfg.data
        data_config.dataloader.test.batch_size = 64
        # data_config.dataset.label_as_caption = True

        data_module = MyDataModule(data_config, processor, augmentation=RandAugment(), num_views=2)
        test_dataloaders = data_module.get_test_dataloaders()
        cifar10_dataloader = test_dataloaders["CIFAR10"]["test"]
        print("Loaded CIFAR10 dataloader")
        print("Computing CKA")

        def f_extract(batch):
            return {"pixel_values": batch[0]}
        
        cka_matrix = cka(cifar10_dataloader, epochs=1, 
                         f_extract=f_extract, f_args=dict(), 
                         model1_foward=model1_forward, model2_foward=model2_forward,)
        
        # Plot the CKA values
        plot_parameters = {
            # "show_ticks_labels": True,
            # "short_tick_labels_splits": 4,
            "use_tight_layout": True,
            "show_half_heatmap": False,
        }

        cka.plot_cka(
            cka_matrix=cka_matrix,
            save_path=f"data/CKA_layers/RistoAle97",
            title=f"{model1_name} vs {model2_name}_pretrained" if use_pretrained_model else f"{model1_name} vs {model2_name}",
            **plot_parameters,
        )

        cka.save(cka_matrix, f"data/CKA_layers/RistoAle97")


if __name__ == "__main__":
    main()
