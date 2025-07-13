import os
import wandb
import yaml

# Set your wandb project and entity
PROJECT_NAME = 'multimodal'
ENTITY_NAME = 'arena-multimodal-lossfns'

# Directory containing the checkpoints
ckpts_dir = "/home/data/bhavin/"
# Initialize wandb API
api = wandb.Api()

# Dictionary to store run information
run_info_dict = {}

# Iterate over the checkpoint files
for model_id in os.listdir(ckpts_dir):
    if model_id == ".DS_Store": continue
    if len(model_id) != 8: continue
    run_id = model_id  # Get the run ID from the checkpoint file name
    try:
        # Access the run information using the run ID
        run = api.run(f"{ENTITY_NAME}/{PROJECT_NAME}/{run_id}")
        
        # Extract the required information
        use_subset = run.config["dataset"]["use_subset"]["value"]
        subset_fraction = run.config["dataset"]["use_subset"]["subset_fraction"]
        if not use_subset:
            subset_fraction = 1
        loss_functions = run.config["loss"]["losses"]
        model_name = ""
        if "contrastive" in loss_functions:
            model_name = "CLIP"
        if "SimCLR" in loss_functions:
            model_name = model_name + " + SimCLR" if model_name else "SimCLR"
        if "image_text_matching" in loss_functions:
            model_name = model_name + " + ITM"
        if "MLM" in loss_functions:
            model_name = model_name + " + MLM"
        image_encoder = run.config["model"]["image_encoder_name"]
        
        # Store the information in the dictionary
        run_info_dict[run_id] = {
            'model': model_name,
            'subset_fraction': subset_fraction,
            'image_encoder': image_encoder
        }
        
        print(f"Successfully retrieved information for run: {run_id}")
    except wandb.errors.CommError as e:
        print(f"Error retrieving information for run {run_id}: {e}")

# Write the collected run information to a YAML file
with open('run_info.yaml', 'w') as f:
    yaml.dump(run_info_dict, f, default_flow_style=False)

print("Run information has been written to 'run_info.yaml'")