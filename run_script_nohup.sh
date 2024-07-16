#!/bin/bash

# Path to the Python script
PYTHON_SCRIPT="/home/phisch/multimodal/train_model_coco_dualenc_new.py"

# Path to the log file
LOG_FILE="/home/phisch/multimodal/train_model_coco_dualenc_new.py"

# Start the Python script with nohup and write output to the log file
nohup python3 "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &
