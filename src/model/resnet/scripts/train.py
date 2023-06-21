import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import loggers

from src.data.tooth_dataset import ToothDataset
from src.model.resnet.resnet import ResNet

if __name__ == "__main__":
    with open("./src/model/resnet/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load train val test splits
    y_train, y_val, y_test = np.load(f"data/final/y_{config['data_type']}_train.npy", allow_pickle=True), \
        np.load(f"data/final/y_{config['data_type']}_val.npy", allow_pickle=True), \
        np.load(f"data/final/y_{config['data_type']}_test.npy", allow_pickle=True)

    dataset_args = dict(image_dir=f"{config['image_dir']}/{config['data_type']}/xrays")
    dataset_train = ToothDataset(y_train, **dataset_args)
    dataset_val = ToothDataset(y_val, **dataset_args)
    dataset_test = ToothDataset(y_test, **dataset_args)

    # Define dataloaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True,
                       collate_fn=ToothDataset.collate_fn)
    loader_train = DataLoader(dataset_train, shuffle=True, **loader_args)
    loader_val = DataLoader(dataset_val, **loader_args)
    loader_test = DataLoader(dataset_test, **loader_args)

    # Define model
    model = ResNet(config)
    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)
    trainer_args = dict(max_epochs=config["max_epochs"],
                        callbacks=[ModelCheckpoint(save_top_k=1,
                                                   monitor="val_loss",
                                                   mode="min",
                                                   filename="epoch={epoch:02d}-val_loss={val_loss:.2f}")],
                        logger=logger)

    # Train model
    if device.type == "cpu":
        trainer = Trainer(**trainer_args)
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=2, **trainer_args)
    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_val)
