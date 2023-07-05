import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.data.tooth_dataset import ToothDataset

if __name__ == "__main__":
    # Load data
    data = np.load("data/processed/y_quadrant_enumeration_disease_with_healthy_samples_unpacked.npy", allow_pickle=True)
    dataset = ToothDataset(data, "data/raw/training_data/quadrant_enumeration_disease/xrays")
    # Save samples
    for i, sample in enumerate(tqdm(dataset, total=len(dataset))):
        image = sample["image"]
        label = sample["label"]
        fig, ax = plt.subplots()
        plt.imshow(image[0], cmap="gray")
        fig.savefig(f"output/tooth_dataset/{label}_{i}_{data[i]['file_name']}")
