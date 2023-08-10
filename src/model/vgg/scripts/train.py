import json

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning import loggers
from torchvision.transforms import transforms, InterpolationMode

from src.data.tooth_dataset import ToothDataset
from src.model.vgg.vgg import Vgg
from src.utils.transforms import SquarePad

if __name__ == "__main__":
    with open("./src/model/vgg/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=10, scale=(0.8, 1.2)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        SquarePad(),
        transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)  # Resize to 256 on the smaller edge
    ])

    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)  # Resize to 256 on the smaller edge
    ])

    # Load train val test splits
    with open("data/final/train_quadrant_enumeration_disease_healthy_unpacked_train.json") as f:
        X_train = json.load(f)
    with open("data/final/train_quadrant_enumeration_disease_healthy_unpacked_val.json") as f:
        X_val = json.load(f)

    if config["n_classes"] > 1:
        # Remove healthy samples
        X_train = list(filter(lambda x: int(x["annotation"]["category_id_3"]) != 4, X_train))
        X_val = list(filter(lambda x: int(x["annotation"]["category_id_3"]) != 4, X_val))

        # Define dataset
        dataset_args = dict(image_dir=f"data/raw/training_data/quadrant_enumeration_disease/xrays",
                            n_classes=config["n_classes"])
        dataset_train = ToothDataset(X_train, transform=transform_train, **dataset_args)
        dataset_val = ToothDataset(X_val, transform=transform, **dataset_args)

        # Prepare weighted sampler for balancing class distribution across epochs
        encoder = LabelEncoder()
        targets = [
            sample["annotation"]["category_id_3"]
            for sample in X_train
        ]
        targets_encoded = encoder.fit_transform(targets)
        class_sample_count = torch.bincount(torch.from_numpy(targets_encoded))
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in targets_encoded])
        sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

        # Define loaders
        loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
        loader_train = DataLoader(dataset_train, sampler=sampler, **loader_args)
        loader_val = DataLoader(dataset_val, **loader_args)
    else:
        # Filter boxes such that only high confidence predicted healthy boxes are included
        X_train = list(filter(lambda x: "score" not in x["annotation"] or x["annotation"]["score"] >= .9, X_train))
        X_val = list(filter(lambda x: "score" not in x["annotation"] or x["annotation"]["score"] >= .9, X_val))

        # Prepare weighted sampler for balancing class distribution across epochs
        encoder = LabelEncoder()
        targets = [
            sample["annotation"]["category_id_3"]
            for sample in X_train
        ]
        targets_encoded = encoder.fit_transform(targets)
        class_sample_count = torch.bincount(torch.from_numpy(targets_encoded))
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in targets_encoded])
        sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

        # In binary classification split labels in healthy vs unhealthy
        for x in X_train:
            if x["annotation"]["category_id_3"] == 4:
                x["annotation"]["category_id_3"] = 0
            else:
                x["annotation"]["category_id_3"] = 1

        for x in X_val:
            if x["annotation"]["category_id_3"] == 4:
                x["annotation"]["category_id_3"] = 0
            else:
                x["annotation"]["category_id_3"] = 1

        # Define dataset
        dataset_args = dict(image_dir=f"data/raw/training_data/quadrant_enumeration_disease/xrays",
                            n_classes=config["n_classes"])
        dataset_train = ToothDataset(X_train, transform=transform_train, **dataset_args)
        dataset_val = ToothDataset(X_val, transform=transform, **dataset_args)

        # Define loaders
        loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
        loader_train = DataLoader(dataset_train, sampler=sampler, **loader_args)
        loader_val = DataLoader(dataset_val, **loader_args)

    # Define model
    model = Vgg(config)
    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)
    trainer_args = dict(max_epochs=config["max_epochs"],
                        callbacks=[ModelCheckpoint(save_top_k=1,
                                                   monitor="val_f1",
                                                   mode="max",
                                                   filename="epoch={epoch:02d}-val_loss={val_f1:.2f}")],
                        logger=logger)

    # Train model
    if device.type == "cpu":
        trainer = Trainer(**trainer_args)
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=2, **trainer_args)
    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_val)
