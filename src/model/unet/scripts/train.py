import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning import loggers
from torchvision.transforms import transforms, InterpolationMode

from src.data.tooth_dataset import ToothDataset
from src.data.tooth_segmentation_dataset import ToothSegmentationDataset
from src.model.unet.unet import UNet
from src.model.vgg.vgg import Vgg
from src.utils.transforms import SquarePad

if __name__ == "__main__":
    with open("./src/model/vgg/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms
    transform_input_train = transforms.Compose([
        transforms.RandomAffine(degrees=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        SquarePad(),
        transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)  # Resize to 256 on the smaller edge
    ])
    transform_input = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)  # Resize to 256 on the smaller edge
    ])
    transform_target = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=InterpolationMode.NEAREST)  # Resize to 256 on the smaller edge
    ])

    # Load train val test splits
    y_train, y_val, y_test = np.load(
        f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_and_segmentation_unpacked_train.npy",
        allow_pickle=True), \
        np.load(f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_and_segmentation_unpacked_val.npy",
                allow_pickle=True), \
        np.load(f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_and_segmentation_unpacked_test.npy",
                allow_pickle=True)

    dataset_args = dict(data_dir=config["data_dir"])
    dataset_train = ToothSegmentationDataset(y_train,
                                             transform_input=transform_input_train,
                                             transform_target=transform_target, **dataset_args)
    dataset_val = ToothSegmentationDataset(y_val,
                                           transform_input=transform_input,
                                           transform_target=transform_target,
                                           **dataset_args)
    dataset_test = ToothSegmentationDataset(y_test,
                                            transform_input=transform_input,
                                            transform_target=transform_target,
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
    model = UNet(config)
    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)
    trainer_args = dict(max_epochs=config["max_epochs"],
                        callbacks=[ModelCheckpoint(save_top_k=1,
                                                   monitor="val_dice_score",
                                                   mode="max",
                                                   filename="epoch={epoch:02d}-val_dice_score={val_dice_score:.2f}")],
                        logger=logger)

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
