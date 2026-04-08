import pickle
import os
import torch
import rootutils
from omegaconf import OmegaConf
from torchvision.transforms import RandAugment
import hydra

ROOT = rootutils.setup_root("/home/phisch/multimodal", indicator=".project-root", pythonpath=True)

from src.analysis.torch_cka import CKA
from src.model.utils import get_model_and_processor
from src.model.model_module import LitMML, MLMWrapper
from src.data.data_module import MyDataModule


@hydra.main(version_base=None, config_path="configs", config_name="cka")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device(f"cuda:{cfg.device[0]}")

    model_config = cfg.model
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
        i = int(len(model_pairs) / 2)
        model_pairs = model_pairs[i:]
        print(f"Using model pairs: {model_pairs}")
    if cfg.get("exclude_model_pairs", None) is not None:
        exclude_pairs = cfg.exclude_model_pairs
        print(f"Excluding model pairs: {exclude_pairs}")
        model_pairs = [pair for pair in model_pairs if pair not in exclude_pairs]
        
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
    
        model1, processor = get_model_and_processor(model_config)
        model1 = MLMWrapper(model1)
        lit_model_1 = LitMML.load_from_checkpoint(ckpt_path_m1, 
                                                model=model1, 
                                                processor=processor,
                                                )
        model1 = lit_model_1.model
        print(f"Loaded model 1 ({model_id_1})")

        if use_pretrained_model:
            lit_model_2 = LitMML(model=model_pretrained,
                               processor=processor,
                               loss_cfg=cfg.loss,
                               optimizer_cfg=cfg.optimizer,
                               scheduler_cfg=cfg.scheduler,
                               )
            model2 = lit_model_2.model
            print(f"Loaded pretrained model 2 ({model_id_2})")
        else:
            model2, _ = get_model_and_processor(model_config)
            model2 = MLMWrapper(model2)
            lit_model_2 = LitMML.load_from_checkpoint(ckpt_path_m2, 
                                                    model=model2, 
                                                    processor=processor,
                                                    )
            model2 = lit_model_2.model
            print(f"Loaded model 2 ({model_id_2})")

        model1_name = cfg.checkpoints[model_id_1].model
        model2_name = cfg.checkpoints[model_id_2].model
        
        vision_layers = [f"basemodel.vision_model.encoder.layer.{i}" for i in range(12)] + ["basemodel.vision_model.pooler", "basemodel.visual_projection"]
        # vision_layers = [f"basemodel.vision_model.encoder.layer.{i}.attention.output" for i in range(12)]
        # vision_layers += [f"basemodel.vision_model.encoder.layer.{i}.intermediate" for i in range(12)]
        # vision_layers += [f"basemodel.vision_model.encoder.layer.{i}.output" for i in range(12)]
        # vision_layers += ["basemodel.visual_projection"]
        text_layers = [f"text_model.encoder.layer.{i}" for i in range(12)] + ["text_model.pooler", "text_projection"]
        # print(layers)
        cka = CKA(model1, model2,
            model1_name = model1_name, model2_name = model2_name,
            # model1_name = model1_name + "(vision layers)", model2_name = model2_name + "(text layers)",
            model1_layers=vision_layers, model2_layers=vision_layers,
            # model1_layers=text_layers, model2_layers=text_layers,
            # model1_layers=vision_layers, model2_layers=text_layers,
            device=device)
        cka.model1 = lambda x: model1.get_image_features(pixel_values=x)
        cka.model2 = lambda x: model2.get_image_features(pixel_values=x)
        # cka.model1 = lambda x: model1.get_text_features(input_ids=x)
        # cka.model2 = lambda x: model2.get_text_features(input_ids=x)
        
        data_config = cfg.data
        data_config.dataloader.test.batch_size = 64
        # data_config.dataset.label_as_caption = True

        

        data_module = MyDataModule(data_config, processor, augmentation=RandAugment(), num_views=2)
        test_dataloaders = data_module.get_test_dataloaders()
        cifar10_dataloader = test_dataloaders["CIFAR10"]["test"]
        print("Loaded CIFAR10 dataloader")
        print("Computing CKA")
        cka.compare(cifar10_dataloader)
        results = cka.export()
        with open(f"data/CKA_layers/full_dataset_aug_mlm/CKA-CIFAR10-{model_id_1}-{model_id_2}.pkl", "wb") as f:
            pickle.dump(results, f)
        cka.plot_results(save_path=f"data/CKA_layers/full_dataset_aug_mlm/CKA-CIFAR10-{model_id_1}-{model_id_2}.png")


if __name__ == "__main__":
    main()
