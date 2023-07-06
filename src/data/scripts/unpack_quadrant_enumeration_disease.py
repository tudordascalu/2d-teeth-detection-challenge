import numpy as np
import torch

if __name__ == "__main__":
    # Load data
    data = np.load("data/processed/y_quadrant_enumeration_disease.npy", allow_pickle=True)

    # Unpack data
    data_unpacked = []
    for sample in data:
        for annotation in sample["annotations"]:
            if torch.is_tensor(annotation["category_id_3"]):
                annotation["category_id_3"] = annotation["category_id_3"].item()
            data_unpacked.append(dict(file_name=sample["file_name"], annotation=annotation))

    # Save data
    np.save("data/processed/y_quadrant_enumeration_disease_unpacked.npy", data_unpacked)