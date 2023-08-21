#!/bin/bash
# Allocate CPU, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=12000
# Allocate GPUs
#SBATCH -p gpu --gres=gpu:titanrtx:2
# Set maximum run time
#SBATCH --time=24:00:00
## Activate environment
#conda activate ml-env
# Train network
echo "Training started.."
python -m src.model.faster_rcnn.scripts.train_disease_detection
echo "Training ended.."
