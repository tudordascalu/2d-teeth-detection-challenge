import os.path

import numpy as np
import torch
import pytorch_lightning as pl
from torch.nn.functional import softmax
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.multi_object_labelling import AssignmentSolver


class MultiObjectLabelingCNN(pl.LightningModule):
    def __init__(self, config):
        super(MultiObjectLabelingCNN, self).__init__()
        self.save_hyperparameters()
        self.config = config

        # Initialize model layers
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, (1, 3), padding="same", bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, (1, 3), padding="same", bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, (1, 3), padding="same", bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 32, (1, 3), padding="same", bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 1, (1, 3), padding="same", bias=False),
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 32, 32 * 32),
            torch.nn.ReLU()
        )

        # Initialize losses
        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCELoss(weight=torch.tensor([0.05, 0.95]))

        # Initialize assignment solver
        self.assignment_solver = AssignmentSolver()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(-1, 32, 32)
        x = (softmax(x, dim=-1) + softmax(x, dim=-2)) / 2
        return x

    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(),
                            lr=float(self.config["learning_rate"]),
                            weight_decay=float(self.config["weight_decay"]),
                            momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=self.config["scheduler_patience"],
                                      verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        accuracy = self._accuracy(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        accuracy = self._accuracy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        accuracy = self._accuracy(y_pred, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, **kwargs):
        x_batch, y_batch, id_batch, sample_batch = batch
        y_batch = y_batch.argmax(-1).cpu().detach().numpy()
        y_pred_batch = self.forward(x_batch)
        y_pred_aligned_batch, y_pred_2d_batch = self.assignment_solver(y_pred_batch.clone().cpu().detach().numpy())
        for y, y_pred, y_pred_aligned, y_pred_2d, id, sample in zip(y_batch, y_pred_batch, y_pred_aligned_batch,
                                                                    y_pred_2d_batch, id_batch, sample_batch):
            if not os.path.exists(f"output/aligner/{id}/"):
                os.mkdir(f"output/aligner/{id}/")
            np.save(f"output/aligner/{id}/labels_{self.config['jaw']}_{sample}", y)
            np.save(f"output/aligner/{id}/labels_pred_activation_{self.config['jaw']}_{sample}", y_pred)
            np.save(f"output/aligner/{id}/labels_pred_2d_{self.config['jaw']}_{sample}", y_pred_2d)
            np.save(f"output/aligner/{id}/labels_pred_{self.config['jaw']}_{sample}", y_pred_aligned)
        return y_pred_batch

    def _loss(self, y_pred, y_true):
        """
        The loss is applied both column-wise and row-wise in order to promote single-class predictions.

        :param y_pred: torch.Tensor of shape (n_batch_size, 17, 17)
        :param y_true: torch.Tensor of shape (n_batch_size, 17, 17)
        :return: torch.Tensor featuring loss value
        """
        # Compute mean squared error (MSE) loss
        a = self.mse_loss(y_pred.reshape(-1, 32), y_true.reshape(-1, 32))

        # Process y_pred and y_true for binary cross entropy (BCE) calculation
        y_pred_processed = y_pred.clone().detach().numpy()
        _, y_pred_processed = self.assignment_solver(y_pred_processed)
        y_pred_processed = torch.from_numpy(y_pred_processed).type(torch.float32)
        y_pred_processed = (y_pred * y_pred_processed).reshape(-1, 1)
        y_pred_processed = torch.stack((1 - y_pred_processed, y_pred_processed), dim=1).squeeze()
        y_true_processed = y_true.reshape(-1, 1)
        y_true_processed = torch.stack((1 - y_true_processed, y_true_processed), dim=1).squeeze()

        # Compute BCE loss, where all other components are zero
        b = self.bce_loss(y_pred_processed, y_true_processed)

        # Get lambda value
        lam = float(self.config["lambda"])

        # Combine MSE and BCE loss using the lambda value
        return (1 - lam) * a + lam * b

    def _accuracy(self, y_pred, y_true):
        """
        Accuracy is computed as the total number of correctly identified teeth.
        We can also consider computing it as total number of correctly identified misaligned teeth, as we have fewer cases.

        :param y_pred: torch.Tensor of shape (n_batch_size, 17, 17)
        :param y_true: torch.Tensor of shape (n_batch_size, 17, 17)
        :return: total number of correctly identified teeth / total number of teeth
        """

        y_true_processed = y_true.argmax(-1).reshape(-1)
        y_pred_processed, _ = self.assignment_solver(y_pred.clone().cpu().detach().numpy())
        y_pred_processed = torch.from_numpy(y_pred_processed).reshape(-1)
        return (y_pred_processed == y_true_processed).sum() / len(y_pred_processed)
