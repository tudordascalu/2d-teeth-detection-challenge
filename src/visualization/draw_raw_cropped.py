import os

import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms import transforms
from tqdm import tqdm

if __name__ == "__main__":
    # Load data
    y = np.load(f"data/final/y_quadrant_enumeration_disease_unpacked_train_filtered.npy", allow_pickle=True)

    # Helpers
    to_pil = transforms.ToPILImage()

    for sample in tqdm(y, total=len(y)):
        # Extract data
        file_name = sample["file_name"]
        box = torch.tensor(sample["annotation"]["bbox"], dtype=torch.int32)
        diseases = ["embedded", "caries", "apical_lesion", "deep_caries"]

        # Extract
        target_disease = sample["annotation"]["category_id_3"]
        target_tooth = sample["annotation"]["category_id_2"]
        target_quadrant = sample["annotation"]["category_id_1"]

        image = read_image(f"data/raw/training_data/quadrant_enumeration_disease/xrays/{file_name}")
        image = image[:, box[1]:box[3], box[0]:box[2]]

        # Save to png
        if not os.path.exists(
                f"visualizations/quadrant_enumeration_disease_filtered_cropped/{diseases[target_disease]}"):
            os.mkdir(f"visualizations/quadrant_enumeration_disease_filtered_cropped/{diseases[target_disease]}")
        image_pil = to_pil(image)
        image_pil.save(
            f"visualizations/quadrant_enumeration_disease_filtered_cropped/{diseases[target_disease]}/{target_disease}_{target_tooth}_{target_quadrant}_{file_name}")
