import json
import os

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from torchvision.transforms import transforms, InterpolationMode
from src.data.tooth_segmentation_dataset import ToothSegmentationDataset
from src.model.unet.unet import UNet
from src.utils.transforms import SquarePad


def mask_exists(x):
    file_name = x["file_name"].split(".")[0]
    category_id_1 = x["annotation"]["category_id_1"]
    category_id_2 = x["annotation"]["category_id_2"]
    category_id_3 = x["annotation"]["category_id_3"]
    return os.path.exists(
        f"data/final/masks/{category_id_3}_{category_id_2}_{category_id_1}_{file_name}_Segmentation.nii")


if __name__ == "__main__":
    with open("./src/model/unet/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=.5),
        transforms.RandomAffine(degrees=15, scale=(0.8, 1.2), translate=(.1, .1), shear=10),
    ])
    transform_input_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        SquarePad(),
        transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)
    ])
    transform_input = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)
    ])
    transform_target = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=InterpolationMode.NEAREST)  # Resize to 256 on the smaller edge
    ])

    # Load train val test splits
    with open("data/final/train_quadrant_enumeration_disease_healthy_unpacked_train.json") as f:
        X_train = json.load(f)
    with open("data/final/train_quadrant_enumeration_disease_healthy_unpacked_val.json") as f:
        X_val = json.load(f)

    # Include only apical lesions for segmentation mode
    X_train = list(filter(lambda x: x["annotation"]["category_id_3"] == 2 and mask_exists(x), X_train))
    X_val = list(filter(lambda x: x["annotation"]["category_id_3"] == 2 and mask_exists(x), X_val))

    dataset_args = dict(data_dir=config["data_dir"], transform_target=transform_target)
    dataset_train = ToothSegmentationDataset(X_train,
                                             transform_input=transform_input_train,
                                             transform=transform_train,
                                             **dataset_args)
    dataset_val = ToothSegmentationDataset(X_val,
                                           transform_input=transform_input,
                                           **dataset_args)
    # Initialize loaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
    loader_train = DataLoader(dataset_train, **loader_args)
    loader_val = DataLoader(dataset_val, **loader_args)

    # Define model
    model = UNet(config)
    model.update_classification_requires_grad(False)
    model.update_segmentation_requires_grad(True)

    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)
    trainer_args = dict(max_epochs=config["max_epochs"],
                        callbacks=[ModelCheckpoint(save_top_k=1,
                                                   monitor="val_loss",
                                                   mode="min",
                                                   filename="epoch={epoch:02d}-val_loss={val_loss:.2f}")],
                        logger=logger,
                        log_every_n_steps=2)

    # Train model
    if device.type == "cpu":
        trainer = Trainer(**trainer_args)
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=1, **trainer_args)

    # Train best model
    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_val)
