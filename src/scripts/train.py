import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from torchvision.transforms import Compose

from src.data.data import PanoramicDataset
from src.model.faster_rcnn import FasterRCNN
from src.utils.transforms import PadToSize

if __name__ == "__main__":
    with open("./src/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_train, y_val, y_test = np.load("data/final/y_quadrant_enumeration_train.npy", allow_pickle=True), \
        np.load("data/final/y_quadrant_enumeration_val.npy", allow_pickle=True), \
        np.load("data/final/y_quadrant_enumeration_test.npy", allow_pickle=True)
    # Define dataset
    # transforms = Compose([
    #     PadToSize(np.array(config["target_size"]).astype(np.int32)),
    # ])
    dataset_args = dict(image_dir=f"{config['image_dir']}/{config['data_type']}/xrays")
    dataset_train = PanoramicDataset(y_train, **dataset_args)
    dataset_val = PanoramicDataset(y_val, **dataset_args)
    dataset_test = PanoramicDataset(y_test, **dataset_args)
    # Define dataloaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True,
                       collate_fn=PanoramicDataset.collate_fn)
    loader_train = DataLoader(dataset_train, shuffle=True, **loader_args)
    loader_val = DataLoader(dataset_val, **loader_args)
    loader_test = DataLoader(dataset_test, **loader_args)
    # Define model
    model = FasterRCNN(config)
    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)
    trainer_args = dict(max_epochs=config["max_epochs"],
                        callbacks=[ModelCheckpoint(save_top_k=2,
                                                   monitor="val_loss",
                                                   mode="min")],
                        logger=logger,
                        log_every_n_steps=100)
    if device.type == "cpu":
        trainer = Trainer(**trainer_args)
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=2, **trainer_args)
    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_val)
