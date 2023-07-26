import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import nibabel as nib
import torch.nn.functional as F

from src.utils.one_hot import OneHotMultiLabel


class ToothSegmentationDataset(Dataset):
    def __init__(self, dataset, data_dir="data", transform=None, transform_input=None, transform_target=None):
        self.dataset = dataset
        self.transform = transform
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.data_dir = data_dir
        self.one_hot_multilabel = OneHotMultiLabel()

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_name = sample['file_name']
        quadrant = sample["annotation"]["category_id_1"]
        tooth = sample["annotation"]["category_id_2"]
        disease = sample["annotation"]["category_id_3"]
        disease_list = sample["annotation"]["category_id_3_list"]

        # Load image, box, mask
        image = read_image(f"{self.data_dir}/raw/training_data/quadrant_enumeration_disease/xrays/{image_name}")
        box = torch.tensor(sample["annotation"]["bbox"], dtype=torch.int32)

        # Crop and normalize image
        image = image[0, box[1]:box[3], box[0]:box[2]].unsqueeze(0) / 255

        # Load mask
        try:
            mask = nib.load(
                f"{self.data_dir}/final/masks/{disease}_{tooth}_{quadrant}_{image_name.split('.')[0]}_Segmentation.nii").get_fdata()
        except:
            mask = np.zeros((image.shape[2], image.shape[1], 1))
        mask = torch.from_numpy(mask).permute(2, 1, 0)

        # Apply transforms
        if self.transform_input is not None:
            image = self.transform_input(image)
        if self.transform_target is not None:
            mask = self.transform_target(mask)
        if self.transform is not None:
            image_mask = torch.concat((image, mask), dim=0)
            image_mask_transformed = self.transform(image_mask)
            image, mask = image_mask_transformed[:1, ...], image_mask_transformed[1:, ...]

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = self.one_hot_multilabel(torch.tensor(disease_list, dtype=torch.int64).unsqueeze(0),
                                        num_classes=5).squeeze()
        mask = torch.tensor(mask, dtype=torch.float32)

        return dict(image=image, label=label, mask=mask, file_name=sample["file_name"])

    def __len__(self):
        return len(self.dataset)
