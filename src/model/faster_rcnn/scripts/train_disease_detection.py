import json

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import loggers

from src.data.panoramic_dataset import PanoramicDataset, PanoramicDiseaseDataset
from src.model.faster_rcnn.faster_rcnn import FasterRCNN

if __name__ == "__main__":
    with open("./src/model/faster_rcnn/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load train val test splits
    with open("data/final/train_quadrant_enumeration_disease_train.json") as f:
        X_train = json.load(f)
    with open("data/final/train_quadrant_enumeration_disease_val.json") as f:
        X_val = json.load(f)

    # Filter samples with no annotations
    X_train = list(filter(lambda x: len(x["annotations"]) > 0, X_train))
    X_val = list(filter(lambda x: len(x["annotations"]) > 0, X_val))

    dataset_args = dict(image_dir=f"data/raw/training_data/quadrant_enumeration_disease/xrays")
    dataset_train = PanoramicDiseaseDataset(X_train, **dataset_args)
    dataset_val = PanoramicDiseaseDataset(X_val, **dataset_args)

    # Define dataloaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True,
                       collate_fn=PanoramicDataset.collate_fn)
    loader_train = DataLoader(dataset_train, shuffle=True, **loader_args)
    loader_val = DataLoader(dataset_val, **loader_args)

    # Define model
    model = FasterRCNN(config)
    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)
    trainer_args = dict(max_epochs=config["max_epochs"],
                        callbacks=[ModelCheckpoint(save_top_k=1,
                                                   monitor="val_loss",
                                                   mode="min",
                                                   filename="epoch={epoch:02d}-val_loss={val_loss:.2f}")],
                        logger=logger,
                        log_every_n_steps=5)

    # Train model
    if device.type == "cpu":
        trainer = Trainer(**trainer_args)
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=2, **trainer_args)
    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_val)
