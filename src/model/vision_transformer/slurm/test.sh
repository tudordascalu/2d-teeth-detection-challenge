#!/bin/bash
# Allocate CPU, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=12000
# Allocate GPUs
#SBATCH -p gpu --gres=gpu:titanrtx:1
# Set maximum run time
#SBATCH --time=24:00:00
## Activate environment
#conda activate ml-env
# Train network
echo "Testing started.."
python -m src.model.vision_transformer.scripts.test
echo "Testing ended.."
