import json

import torch
import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from torchvision.transforms import transforms, InterpolationMode
from src.data.tooth_segmentation_dataset import ToothSegmentationDataset
from src.model.unet.unet import UNet
from src.utils.transforms import SquarePad

if __name__ == "__main__":
    with open("./src/model/unet/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms
    transform_input = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)
    ])
    transform_target = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=InterpolationMode.NEAREST)  # Resize to 256 on the smaller edge
    ])

    # Load train val test splits
    with open("data/final/train_quadrant_enumeration_disease_healthy_unpacked_test.json") as f:
        X_test = json.load(f)

    dataset_args = dict(data_dir=config["data_dir"], transform_target=transform_target)
    dataset_test = ToothSegmentationDataset(X_test, transform_input=transform_input, **dataset_args)

    # Initialize loaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
    loader_test = DataLoader(dataset_test, **loader_args)

    # Define model
    model = UNet.load_from_checkpoint(config["checkpoint_path"], config=config, map_location=device)

    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)

    # Train model
    if device.type == "cpu":
        trainer = Trainer()
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=1)

    # After training, load the best model according to validation loss
    trainer.test(model, dataloaders=loader_test)
