import json

import torch
import yaml
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.nn.functional as F

from src.data.tooth_segmentation_dataset import ToothSegmentationDataset
from src.model.unet.unet import UNet
from src.utils.transforms import SquarePad
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("./src/model/unet/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    with open("data/final/train_quadrant_enumeration_disease_healthy_unpacked_test.json") as f:
        X = json.load(f)

    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR)
    ])
    to_pil = transforms.ToPILImage()

    dataset = ToothSegmentationDataset(
        X,
        transform=transform
    )

    # Load model
    model = UNet.load_from_checkpoint(
        f"checkpoints/unet/version_53/checkpoints/epoch=epoch=08-val_loss=val_loss=0.85.ckpt"
    )
    model.eval()

    # Predict for each sample
    for sample in tqdm(dataset, total=len(dataset)):
        input = sample["image"]
        file_name = sample["file_name"]
        label = sample["label"]

        with torch.no_grad():
            prediction = model(input.unsqueeze(0))
        prediction_segmentation = F.sigmoid(prediction["segmentation"]).squeeze()
        prediction_segmentation = to_pil(prediction_segmentation)
        plt.imshow(prediction_segmentation, cmap="gray")
        plt.show()
        plt.close()
        # Save prediction using combination of disease and bounding box coordinates in order to be able to identify
        # prediction.save(f"output/unet/{disease}_{box[0]}{box[1]}{box[2]}{box[3]}_{file_name}")
