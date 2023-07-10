import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import nibabel as nib


class ToothSegmentationDataset(Dataset):
    def __init__(self, dataset, data_dir="data", transform_input=None, transform_target=None):
        self.dataset = dataset
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.data_dir = data_dir

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_name = sample['file_name']
        disease = sample["annotation"]["category_id_3"]

        # Load image, box, mask
        image = read_image(f"{self.data_dir}/raw/training_data/quadrant_enumeration_disease/xrays/{image_name}")
        box = torch.tensor(sample["annotation"]["bbox"], dtype=torch.int32)

        # Crop and normalize image
        image = image[:, box[1]:box[3], box[0]:box[2]] / 255

        # Load mask
        try:
            quadrant = sample["annotation"]["category_id_1"]
            tooth = sample["annotation"]["category_id_2"]
            mask = nib.load(
                f"{self.data_dir}/final/masks/{disease}_{tooth}_{quadrant}_{image_name.split('.')[0]}_Segmentation.nii").get_fdata()
        except:
            # Create empty mask if it does not exist
            mask = np.zeros((image.shape[2], image.shape[1]))
        mask = torch.from_numpy(mask).permute(2, 1, 0).type(torch.float32)

        # Apply transforms
        if self.transform_input is not None:
            image = self.transform_input(image)
        if self.transform_target is not None:
            mask = self.transform_target(mask)

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(disease, dtype=torch.int64)
        mask = torch.tensor(mask, dtype=torch.float32)

        return dict(image=image, label=label, mask=mask, file_name=sample["file_name"])

    def __len__(self):
        return len(self.dataset)
