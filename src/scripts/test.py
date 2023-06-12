import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.data.data import PanoramicDataset
from src.model.faster_rcnn import FasterRCNN

if __name__ == "__main__":
    with open("./src/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)
    checkpoint = dict(version="version_3", model="epoch=76-step=2156.ckpt")
    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load test split
    y_test = np.load("data/final/y_quadrant_enumeration_test.npy", allow_pickle=True)
    dataset_args = dict(image_dir=f"{config['image_dir']}/{config['data_type']}/xrays", data_type=config["data_type"])
    dataset_test = PanoramicDataset(y_test, **dataset_args)
    # Define dataloader
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True,
                       collate_fn=PanoramicDataset.collate_fn)
    loader_test = DataLoader(dataset_test, **loader_args)
    # Define model
    model = FasterRCNN.load_from_checkpoint(f"checkpoints/{checkpoint['version']}/checkpoints/{checkpoint['model']}")
    if device.type == "cpu":
        trainer = Trainer()
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=2)
    trainer.test(model, dataloaders=loader_test)
