import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from torchvision.transforms import Compose

from src.data.multi_object_dataset import MultiObjectDataset
from src.model.multi_object_labeling_cnn.multi_object_labeling_cnn import MultiObjectLabelingCNN
from src.utils.transforms import RandomObjectRemover, RandomObjectSwapper, RandomObjectShifter

if __name__ == "__main__":
    with open("src/model/multi_object_labeling_cnn/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load train val test splits
    y_train, y_val, y_test = np.load(f"data/final/y_quadrant_enumeration_train.npy", allow_pickle=True), \
        np.load(f"data/final/y_quadrant_enumeration_val.npy", allow_pickle=True), \
        np.load(f"data/final/y_quadrant_enumeration_test.npy", allow_pickle=True)
    inter_object_distance_mat_mean = np.load("data/final/inter_object_distance_mat_mean.npy")
    inter_object_distance_mat_std = np.load("data/final/inter_object_distance_mat_std.npy")

    # Setup transforms
    transforms = Compose(transforms=[RandomObjectRemover(config["p_remove"], config["max_remove"]),
                                     RandomObjectSwapper(config["p_swap"], config["max_swap"]),
                                     RandomObjectShifter(config["p_shift"], config["max_dist_shift"],
                                                         config["max_count_shift"])])
    dataset_args = dict(inter_object_distance_mat_mean=inter_object_distance_mat_mean,
                        inter_object_distance_mat_std=inter_object_distance_mat_std)
    dataset_train = MultiObjectDataset(dataset=y_train, transforms=transforms, **dataset_args)
    dataset_val = torch.load("data/final/multi_object_dataset_val.pt")
    # Define dataloaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
    loader_train = DataLoader(dataset_train, shuffle=True, **loader_args)
    loader_val = DataLoader(dataset_val, **loader_args)

    # Define model
    model = MultiObjectLabelingCNN(config)
    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)
    trainer_args = dict(max_epochs=config["max_epochs"],
                        callbacks=[ModelCheckpoint(save_top_k=2,
                                                   monitor="val_loss",
                                                   mode="min",
                                                   filename="epoch={epoch:02d}-val_loss={val_loss:.2f}")],
                        logger=logger,
                        log_every_n_steps=5)

    # Start trainer
    if device.type == "cpu":
        trainer = Trainer(**trainer_args)
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=1, **trainer_args)

    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_val)
