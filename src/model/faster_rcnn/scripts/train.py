import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
import albumentations as A

from src.data.panoramic_dataset import PanoramicDataset
from src.model.faster_rcnn.faster_rcnn import FasterRCNN

if __name__ == "__main__":
    with open("./src/model/faster_rcnn/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load train val test splits
    y_train, y_val, y_test = np.load(f"data/final/y_{config['data_type']}_train.npy", allow_pickle=True), \
        np.load(f"data/final/y_{config['data_type']}_val.npy", allow_pickle=True), \
        np.load(f"data/final/y_{config['data_type']}_test.npy", allow_pickle=True)

    # Create transforms
    transform = A.Compose([
        A.HorizontalFlip(p=config["p_flip"]),
        A.RandomBrightnessContrast(p=config["p_brightness_contrast"]),
        A.ShiftScaleRotate(p=config["p_rigid_body"],
                           shift_limit=config["shift_limit"],
                           rotate_limit=config["rotate_limit"],
                           scale_limit=config["scale_limit"]),
        A.Cutout(num_holes=config["num_holes"],
                 max_h_size=config["max_h_size"],
                 max_w_size=config["max_w_size"],
                 fill_value=config["fill_value"],
                 p=config["p_cutout"]),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

    dataset_args = dict(image_dir=f"{config['image_dir']}/{config['data_type']}/xrays", data_type=config["data_type"])
    dataset_train = PanoramicDataset(y_train, transform=transform, **dataset_args)
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
                        callbacks=[ModelCheckpoint(save_top_k=1,
                                                   monitor="loss/val",
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
