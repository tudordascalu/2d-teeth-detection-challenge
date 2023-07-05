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
    checkpoint = dict(version="version_23", model="epoch=epoch=04-val_loss=val_f1=0.31.ckpt")

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load train val test splits
    y_test = np.load(f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_unpacked_test.npy",
                     allow_pickle=True)

    dataset_args = dict(image_dir=f"{config['image_dir']}/{config['data_type']}/xrays")
    dataset_test = ToothDataset(y_test, **dataset_args)

    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True,
                       collate_fn=ToothDataset.collate_fn)
    loader_test = DataLoader(dataset_test, **loader_args)

    # Define model
    model = ResNet.load_from_checkpoint(
        f"checkpoints/resnet/{checkpoint['version']}/checkpoints/{checkpoint['model']}")
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
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=1, **trainer_args)
    trainer.test(model=model, dataloaders=loader_test)
