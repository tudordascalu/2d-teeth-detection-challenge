import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning import loggers

from src.data.tooth_dataset import ToothDataset
from src.model.resnet.resnet import ResNet
from src.model.vision_transformer.vision_transformer import VisionTransformer

if __name__ == "__main__":
    with open("./src/model/resnet/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load train val test splits
    y_train, y_val, y_test = np.load(
        f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_unpacked_train.npy", allow_pickle=True), \
        np.load(f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_unpacked_val.npy", allow_pickle=True), \
        np.load(f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_unpacked_test.npy", allow_pickle=True)

    dataset_args = dict(image_dir=f"{config['image_dir']}/{config['data_type']}/xrays")
    dataset_train = ToothDataset(y_train, **dataset_args)
    dataset_val = ToothDataset(y_val, **dataset_args)
    dataset_test = ToothDataset(y_test, **dataset_args)

    # Prepare weighted sampler for balancing class distribution across epochs
    y_train_labels = torch.tensor([sample["annotation"]["category_id_3"] for sample in y_train], dtype=torch.int64)
    class_sample_count = torch.bincount(y_train_labels)
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in y_train_labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    # Initialize loaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True,
                       collate_fn=ToothDataset.collate_fn)
    loader_train = DataLoader(dataset_train, sampler=sampler, **loader_args)
    loader_val = DataLoader(dataset_val, **loader_args)
    loader_test = DataLoader(dataset_test, **loader_args)

    # Define model
    model = VisionTransformer(config)
    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)
    trainer_args = dict(max_epochs=config["max_epochs"],
                        callbacks=[ModelCheckpoint(save_top_k=1,
                                                   monitor="loss/val",
                                                   mode="min",
                                                   filename="epoch={epoch:02d}-val_loss={val_loss:.2f}")],
                        logger=logger)

    # Train model
    if device.type == "cpu":
        trainer = Trainer(**trainer_args)
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=2, **trainer_args)
    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_val)
