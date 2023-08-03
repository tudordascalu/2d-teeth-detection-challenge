import json

import torch
import yaml
from torchvision.transforms import transforms
from tqdm import tqdm

from src.data.tooth_segmentation_dataset import ToothSegmentationDataset
from src.model.unet.unet import UNet
from src.utils.transforms import SquarePad

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
        f"checkpoints/unet/version_66/checkpoints/epoch=epoch=20-val_loss=val_loss=0.23.ckpt",
        map_location=device
    )
    model.eval()

    # Predict for each sample
    prediction_classification_list = []
    for sample in tqdm(dataset, total=len(dataset)):
        input = sample["image"]
        file_name = sample["file_name"]
        label = sample["label"]

        with torch.no_grad():
            prediction = model(input.unsqueeze(0).to(device))
        prediction_classification_list.append(dict(
            file_name=file_name,
            label=label.cpu().tolist(),
            prediction=prediction["classification"].detach().cpu().tolist()))

    with open(f"output/unet_predict.json", "w") as f:
        json.dump(prediction_classification_list, f)
