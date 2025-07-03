from src.model.utils import get_model_and_processor
from src.model.model_module import LitMML
import os
from omegaconf import OmegaConf

model_config = OmegaConf.load("configs/model/dual_encoder.yaml")
# model_config["model"]["image_encoder_name"] = "google/vit-base-patch16-384"
print(model_config)
model, processor = get_model_and_processor(model_config)
print(model)
ckpts_dir = "/home/data/bhavin/ckpts"

skip_models = [".DS_Store", '2udboe06', 'nq3c74wi', '8estzlel', 'd4112qkb', 'v8r6ibfz', 'bz3dvkm5', 'e2kwl0iu', 'qkk0yi2q', 'zathvtrx', '3b3zvcnp', '2k0xgnrt', 'u5l6rwc4', 'udtu05nw', '6z9rakfu', 'x7kjrc29', 'ps81urf1', 'xarqrl5t', '1j1wb3o6', 'yh1adr3g', '6gvnn76i', '5ib9fce5', '32yprt3g', '8el3y1x8', 'oc0g8fql', '9v1wy0vb', '9nvg456i', '95qxia4w', 'dspp551hg']
skip_models = []
take_models = ["6gvnn76i", "kx1devsu", "qkk0yi2q", "x7kjrc29", "03avtdyk", "08q77hgf"]

for i, model_name in enumerate(os.listdir(ckpts_dir)):
    if model_name in skip_models: 
    # if model_name not in take_models:
        print("Skipping model: ", model_name)
        continue
    files = os.listdir(os.path.join(ckpts_dir, model_name))
    if len(files) == 0:
        print("No checkpoints found for model: ", model_name)
    elif len(files) > 1:
        print("Multiple checkpoints found for model: ", model_name)
    else:
        ckpt_path = os.path.join(ckpts_dir, model_name, files[0])
        print("Loading checkpoint for model: ", model_name)
        # lit_model = LitMML.load_from_checkpoint(ckpt_path, model=model, processor=processor)
        try:
            lit_model = LitMML.load_from_checkpoint(ckpt_path, model=model, processor=processor)
            print("Successfully loaded checkpoint for model: ", model_name)
        except Exception as e:
            print("Error: ", e)
