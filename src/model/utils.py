from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    ViTConfig,
    BertConfig,
    VisionTextDualEncoderConfig,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor
)

# Standard default configuration (based on configs/model/dual_encoder.yaml)
DEFAULT_IMAGE_ENCODER = "google/vit-base-patch16-224"
DEFAULT_TEXT_ENCODER = "google-bert/bert-base-uncased"
# DEFAULT_TOKENIZER_CONFIG = {"use_fast": False, "padding": "max_length", "truncation": True, "max_length": 512}
DEFAULT_TOKENIZER_CONFIG = {"use_fast": False}

def get_model_and_processor(
    config=None, 
    pretrained=False,
    image_encoder_name=None,
    text_encoder_name=None,
    tokenizer_config=None
):
    """
    Loads a VisionTextDualEncoderModel and its processor.
    
    Args:
        config: Optional config object with model attributes (for backward compatibility)
        pretrained: If True, loads pretrained weights; else, initializes from config
        image_encoder_name: Name of the image encoder model (defaults to ViT-base)
        text_encoder_name: Name of the text encoder model (defaults to DistilBERT)
        tokenizer_config: Configuration dict for the tokenizer
    
    Returns:
        tuple: (model, processor)
    """
    # Use parameters if provided, otherwise fall back to config, then defaults
    if config is not None:
        img_encoder = image_encoder_name or config.model.image_encoder_name
        txt_encoder = text_encoder_name or config.model.text_encoder_name
        tok_config = tokenizer_config or getattr(config.model, 'tokenizer', DEFAULT_TOKENIZER_CONFIG)
    else:
        img_encoder = image_encoder_name or DEFAULT_IMAGE_ENCODER
        txt_encoder = text_encoder_name or DEFAULT_TEXT_ENCODER
        tok_config = tokenizer_config or DEFAULT_TOKENIZER_CONFIG
    
    if pretrained:
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            img_encoder,
            txt_encoder
        )
    else:
        vision_config = ViTConfig.from_pretrained(img_encoder)
        text_config = BertConfig.from_pretrained(txt_encoder)
        model_config = VisionTextDualEncoderConfig.from_vision_text_configs(
            vision_config=vision_config, text_config=text_config
        )
        model = VisionTextDualEncoderModel(config=model_config)

    image_processor = AutoImageProcessor.from_pretrained(
        img_encoder, input_data_format="channels_last"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        txt_encoder, **tok_config
    )
    processor = VisionTextDualEncoderProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    
    return model, processor