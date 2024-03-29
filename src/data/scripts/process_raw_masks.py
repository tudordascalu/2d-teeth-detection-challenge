import glob

from tqdm import tqdm
import nibabel as nib
import numpy as np

if __name__ == "__main__":
    masks = glob.glob("data/raw/masks/*")
    for mask in tqdm(masks, total=len(masks)):
        mask_name = mask.split("/")[-1]
        disease = mask_name.split("_")[0]
        tooth = mask_name.split("_")[1]
        quadrant = mask_name.split("_")[2]
        id = mask_name.split("_")[4]
        target = nib.load(mask)

        # Save everything that is binary
        if np.array_equal(np.unique(target.get_fdata()), [0, 1]):
            nib.save(target, f"data/final/masks/{disease}_{tooth}_{quadrant}_train_{id}_Segmentation.nii")
