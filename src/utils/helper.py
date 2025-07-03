from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    ViTConfig,
    BertConfig,
    VisionTextDualEncoderConfig,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor
)

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