import os

def rename_files_in_directory(base_directory, rename_function):
    """
    Recursively renames all files in a directory and its subdirectories.

    :param base_directory: The root directory to start renaming files.
    :param rename_function: A function that takes a filename and returns the new filename.
    """
    for root, _, files in os.walk(base_directory):
        for file in files:
            # if "full_dataset_aug_mlm" in root:
                # continue
            old_path = os.path.join(root, file)
            new_name = rename_function(file)
            new_path = os.path.join(root, new_name)

            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

def example_rename_function(filename):
    """
    Example renaming function that adds a '_renamed' suffix before the file extension.

    :param filename: The original filename.
    :return: The new filename.
    """
    name, ext = os.path.splitext(filename)
    name = name.replace("vis_enc", "vis_proj")

    return f"{name}{ext}"

if __name__ == "__main__":
    # Replace '/path/to/directory' with the directory you want to process
    base_directory = "/home/phisch/multimodal/data/representations/CIFAR10/full_dataset_mlm"
    
    # Call the renaming function
    rename_files_in_directory(base_directory, example_rename_function)