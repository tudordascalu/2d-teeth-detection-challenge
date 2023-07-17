import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning import loggers
from torchvision.transforms import transforms, InterpolationMode
from src.data.tooth_segmentation_dataset import ToothSegmentationDataset
from src.model.unet.unet import UNet
from src.utils.transforms import SquarePad

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
    if config["mode"] == "segmentation":
        y_train, y_val, y_test = np.load(
            f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_and_segmentation_unpacked_train.npy",
            allow_pickle=True), \
            np.load(f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_and_segmentation_unpacked_val.npy",
                    allow_pickle=True), \
            np.load(
                f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_and_segmentation_unpacked_test.npy",
                allow_pickle=True)
    else:
        y_train, y_val, y_test = np.load(
            f"data/final/y_quadrant_enumeration_disease_unpacked_train.npy",
            allow_pickle=True), \
            np.load(f"data/final/y_quadrant_enumeration_disease_unpacked_val.npy",
                    allow_pickle=True), \
            np.load(f"data/final/y_quadrant_enumeration_disease_unpacked_test.npy",
                    allow_pickle=True)

    dataset_args = dict(data_dir=config["data_dir"], transform_target=transform_target)
    dataset_train = ToothSegmentationDataset(y_train,
                                             transform_input=transform_input_train,
                                             transform=transform_train,
                                             **dataset_args)
    dataset_val = ToothSegmentationDataset(y_val,
                                           transform_input=transform_input,
                                           **dataset_args)
    dataset_test = ToothSegmentationDataset(y_test,
                                            transform_input=transform_input,
                                            **dataset_args)

    # Prepare weighted sampler for balancing class distribution across epochs
    encoder = LabelEncoder()
    targets = [
        sample["annotation"]["category_id_3"]
        for sample in y_train
    ]
    targets_encoded = encoder.fit_transform(targets)
    class_sample_count = torch.bincount(torch.from_numpy(targets_encoded))
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets_encoded])
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    # Initialize loaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
    loader_train = DataLoader(dataset_train, sampler=sampler, **loader_args)
    loader_val = DataLoader(dataset_val, **loader_args)
    loader_test = DataLoader(dataset_test, **loader_args)

    # Define model
    if config["checkpoint_path"] is not None:
        model = UNet.load_from_checkpoint(config["checkpoint_path"])
        model.config = config
        if config["mode"] == "segmentation":
            model.update_segmentation_requires_grad(True)
            model.update_classification_requires_grad(False)
        else:
            model.update_segmentation_requires_grad(False)
            model.update_classification_requires_grad(True)
    else:
        model = UNet(config)
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
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=2, **trainer_args)

    # Train best model
    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_val)

    # After training, load the best model according to validation loss
    best_model = UNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(best_model, dataloaders=loader_test)
