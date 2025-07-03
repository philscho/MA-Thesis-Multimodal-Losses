#!/bin/bash

# Run the first script
# python src/analysis/eval_linear_probe_per_layer.py dataset="CIFAR10" checkpoints=full_dataset_im384 lightning.trainer.devices=[2] model.model.image_encoder_name="google/vit-base-patch16-384" data.dataloader.test.batch_size=64

# Run the second script with different parameters after the first one finishes
# python src/analysis/eval_linear_probe_per_layer.py dataset="CIFAR10" checkpoints=full_dataset lightning.trainer.devices=[2]

python src/analysis/eval_linear_probe_per_layer.py dataset="Caltech101" checkpoints=full_dataset lightning.trainer.devices=[3]
