import os
import pickle
import csv

# Directory containing pickle files
input_dir = '/home/phisch/multimodal/test_results/full_dataset_aug_mlm/linear_per_layer'
output_csv = 'test_results/linear_probing_layers_results.csv'

rows = []

for fname in os.listdir(input_dir):
    if fname.startswith('CIFAR10'):
        if fname.endswith('.p') or fname.endswith('.pkl'):
            model_id = fname.split('-')[1]
            dataset = fname.split('-')[0]
            with open(os.path.join(input_dir, fname), 'rb') as f:
                data = pickle.load(f)
                for layer, classifiers in data.items():
                    for clf, metrics in classifiers.items():
                        for topx, value in metrics.items():
                            rows.append({
                                'model_id': model_id,
                                'dataset': dataset,
                                'layer': layer,
                                'metric': topx,
                                'score': value
                            })

with open(output_csv, 'a', newline='') as csvfile:
    fieldnames = ['model_id', 'dataset', 'layer', 'metric', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Saved results to {output_csv}")