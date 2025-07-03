import os

def get_leaf_nodes(directory):
    leaf_nodes = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            leaf_nodes.append(os.path.join(root, file))
    return leaf_nodes

def write_to_file(leaf_nodes, output_file):
    with open(output_file, 'a') as f:
        for node in leaf_nodes:
            f.write(node + '\n')

if __name__ == "__main__":
    directory = "/home/data/arena-multimodal/mlm_ckpts/"  # Replace with your directory path
    output_file = "mlm_checkpoint_paths.txt"
    leaf_nodes = get_leaf_nodes(directory)
    write_to_file(leaf_nodes, output_file)
    print(f"Leaf node paths have been written to {output_file}")