import yaml

def load_model_mapping(mapping_file):
    with open(mapping_file, 'r') as file:
        return yaml.safe_load(file)

def process_checkpoint_paths(checkpoint_file, mapping, output_file):
    with open(checkpoint_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'a') as file:
        for line in lines:
            path = line.strip()
            model_id = path.split('/')[5]
            if not model_id in mapping:
                model_id = path.split('/')[4]
            model_info = mapping[model_id]
            name = model_info['name']
            name = name.replace(' ', '').replace('+', '-')
            subset_fraction = model_info['subset_fraction']
            if model_info['image_encoder'] == 'google/vit-base-patch16-384':
                subset_fraction = "1*"
            file.write(f"{name}_{subset_fraction},{path}\n")

if __name__ == "__main__":
    mapping_file = "/home/phisch/multimodal/configs/checkpoints/model_id_mapping.yaml"
    checkpoint_file = "/home/phisch/multimodal/mlm_checkpoint_paths.txt"
    output_file = "/home/phisch/multimodal/mlm_checkpoint_paths__.txt"

    model_mapping = load_model_mapping(mapping_file)
    process_checkpoint_paths(checkpoint_file, model_mapping, output_file)
    print(f"Updated checkpoint paths have been written to {output_file}")