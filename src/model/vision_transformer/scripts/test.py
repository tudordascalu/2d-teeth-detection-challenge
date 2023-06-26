import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning import loggers
from torchvision.transforms import Compose, Resize, InterpolationMode, CenterCrop

from src.data.tooth_dataset import ToothDataset
from src.model.vision_transformer.vision_transformer import VisionTransformer

if __name__ == "__main__":
    # Load config
    with open("./src/model/vision_transformer/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)
    checkpoint = dict(version="version_2", model="epoch=epoch=03-val_loss=val_loss=0.00.ckpt")

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load train val test splits
    y_test = np.load(f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_unpacked_test.npy",
                     allow_pickle=True)

    transform = Compose([
        Resize(256, interpolation=InterpolationMode.BILINEAR),  # Resize to 256 on the smaller edge
        CenterCrop(224),  # Perform a center crop of 224x224
    ])

    dataset_args = dict(image_dir=f"{config['image_dir']}/{config['data_type']}/xrays", transform=transform)
    dataset_test = ToothDataset(dataset=y_test,
                                image_dir=f"{config['image_dir']}/{config['data_type']}/xrays",
                                transform=transform)

    # Prepare weighted sampler for balancing class distribution across epochs
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True,
                       collate_fn=ToothDataset.collate_fn)
    loader_test = DataLoader(dataset_test, **loader_args)

    # Define model
    model = VisionTransformer.load_from_checkpoint(
        f"checkpoints/vision_transformer/{checkpoint['version']}/checkpoints/{checkpoint['model']}")
    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)
    trainer_args = dict(max_epochs=config["max_epochs"],
                        callbacks=[ModelCheckpoint(save_top_k=1,
                                                   monitor="loss/val",
                                                   mode="min",
                                                   filename="epoch={epoch:02d}-val_loss={loss/val:.2f}")],
                        logger=logger)

    # Train model
    if device.type == "cpu":
        trainer = Trainer(**trainer_args)
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=1, **trainer_args)
    trainer.test(model=model, dataloaders=loader_test)
