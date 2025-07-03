import subprocess

# List of commands to run
commands = [
    "python src/test.py custom_run=ImageNet-linear_probe checkpoints=0.8_dataset lightning.trainer.devices=[3]",
    "python src/test.py custom_run=ImageNet-linear_probe checkpoints=0.6_dataset lightning.trainer.devices=[3]",
    "python src/test.py custom_run=ImageNet-linear_probe checkpoints=0.4_dataset lightning.trainer.devices=[3]",
    "python src/test.py custom_run=ImageNet-linear_probe checkpoints=0.2_dataset lightning.trainer.devices=[3]",
    "python src/test.py custom_run=ImageNet-linear_probe checkpoints=full_dataset_im384 model=dual_encoder_vit-384 lightning.trainer.devices=[3]",
    # Add more commands as needed
]

# Run each command one after the other
for command in commands:
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"Command failed: {command}")
        break