import os
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    ViTConfig,
    BertConfig,
    VisionTextDualEncoderConfig,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor
)

from src.model.model_module import LitMML

def get_model_and_processor(config):
    model_config = VisionTextDualEncoderConfig.from_vision_text_configs(
            vision_config=ViTConfig(), text_config=BertConfig()
    )
    model = VisionTextDualEncoderModel(
        config = model_config,
    )
    # model = VisionTextDualEncoderModel.from_vision_text_pretrained(config.model.image_encoder_name, config.model.text_encoder_name)
    image_processor = AutoImageProcessor.from_pretrained(
            config.model.image_encoder_name,
            input_data_format="channels_last",
    )
    tokenizer = AutoTokenizer.from_pretrained(
            config.model.text_encoder_name, 
            **config.model.tokenizer,
    )
    processor = VisionTextDualEncoderProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    
    return model, processor

def load_lit_model_checkpoint(model_id, model, processor, device):
    if model_id == "n5trwpk9":
        ckpt_path = "/home/data/bhavin/n5trwpk9/ckpt-epoch=34-loss-val=4.378.ckpt"
    elif model_id == "zathvtrx":
        ckpt_path = "/data/bhavin/ckpts_old/zathvtrx/ckpt-epoch=14-loss-val=0.002.ckpt"
    else:
        ckpt_dir = os.path.join(cfg.paths.ckpts_dir, model_id)
        ckpt_files = os.listdir(ckpt_dir)
        if len(ckpt_files) > 1 and "last.ckpt" in ckpt_files:
            ckpt_files.remove("last.ckpt")
        ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])

    lit_model = LitMML.load_from_checkpoint(
        ckpt_path,
        model=model,
        processor=processor,
        map_location=device
    )
    return lit_model