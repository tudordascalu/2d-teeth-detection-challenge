import numpy as np
import torch
from torchvision.transforms import Compose

from src.data.multi_object_dataset import MultiObjectDataset
from src.utils.transforms import RandomObjectRemover, RandomObjectSwapper, RandomObjectShifter

if __name__ == "__main__":
    # Load train val test splits
    y_val = np.load(f"data/final/y_quadrant_enumeration_val.npy", allow_pickle=True)
    inter_object_distance_mat_mean = np.load("data/final/inter_object_distance_mat_mean.npy")
    inter_object_distance_mat_std = np.load("data/final/inter_object_distance_mat_std.npy")

    # Setup transforms
    transforms = Compose(transforms=[RandomObjectRemover(.5, 3),
                                     RandomObjectSwapper(.5, 4),
                                     RandomObjectShifter(.5, 10, 5)])

    # Initialize dataset
    dataset_val = MultiObjectDataset(dataset=y_val,
                                     transforms=transforms,
                                     inter_object_distance_mat_mean=inter_object_distance_mat_mean,
                                     inter_object_distance_mat_std=inter_object_distance_mat_std)

    # Save augmented samples
    dataset_val = [sample for sample in dataset_val]
    torch.save(dataset_val, "data/final/multi_object_dataset_val.pt")
