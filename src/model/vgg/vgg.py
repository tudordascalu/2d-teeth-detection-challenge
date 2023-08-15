import torch.nn
import torchmetrics
from torch import nn
from torch.nn import Linear, BCEWithLogitsLoss
from torchvision import models
import pytorch_lightning as pl
from torchvision.models import VGG16_Weights

from src.model.unet.unet import UNet


class Vgg(pl.LightningModule):
    def __init__(self, config=dict(pretrained=False, n_classes=1, learning_rate=1e-4)):
        super().__init__()
        self.__dict__.update(locals())

        # Hyperparameters
        self.save_hyperparameters()
        self.config = config

        # Metrics
        # TODO: implement multi label solution for
        if self.config["n_classes"] > 1:
            self.train_f1 = torchmetrics.F1Score("multilabel",
                                                 average="macro",
                                                 num_labels=config["n_classes"])
            self.val_f1 = torchmetrics.F1Score("multilabel",
                                               average="macro",
                                               num_labels=config["n_classes"])
            self.val_f1_per_class = torchmetrics.F1Score("multilabel",
                                                         average=None,
                                                         num_labels=config["n_classes"])
            self.test_f1 = torchmetrics.F1Score("multilabel",
                                                average="macro",
                                                num_labels=config["n_classes"])
            self.test_f1_per_class = torchmetrics.F1Score("multilabel",
                                                          average=None,
                                                          num_labels=config["n_classes"])
        else:
            self.train_f1 = torchmetrics.F1Score("binary", average="macro")
            self.val_f1 = torchmetrics.F1Score("binary", average="macro")
            self.test_f1 = torchmetrics.F1Score("binary", average="macro")
        self.bce = BCEWithLogitsLoss()

        # Model
        if "pretrained" in self.config and self.config["pretrained"]:
            vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
            unet = UNet.load_from_checkpoint(self.config["unet_checkpoint"])
        else:
            vgg16 = models.vgg16(weights=None)
            unet = UNet()

        # Prepare encoders
        self.vgg16_encoder = vgg16.features
        self.vgg16_pool = vgg16.avgpool
        self.unet_encoder = nn.Sequential(
            unet.inc,
            unet.down1,
            unet.down2,
            unet.down3,
            unet.down4
        )
        self.unet_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Prepare decoder
        self.classifier = vgg16.classifier
        self.classifier[0] = Linear(self.classifier[0].in_features + 1024 * 7 * 7, self.classifier[0].out_features)
        self.classifier[6] = Linear(self.classifier[6].in_features, config["n_classes"])

        # Freeze all unet layers
        for param in self.unet_encoder.parameters():
            param.requires_grad = False

        # Unfreeze last convolutional blocks in vgg16
        for param in self.vgg16_encoder[:24].parameters():
            param.requires_grad = False

    def forward(self, x):
        # Vgg
        x_vgg = self.vgg16_encoder(x)
        x_vgg = self.vgg16_pool(x_vgg)
        # Unet
        x_unet = self.unet_encoder(
            x[:, :1, ...])  # TODO: select only first dimension from each input image as unet requires grayscale
        x_unet = self.unet_pool(x_unet)

        # Classifier
        x = torch.cat((x_vgg, x_unet), dim=1)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=float(self.config["learning_rate"]))

        # Reduce learning rate when a metric has stopped improving.
        # "val_loss" is the logged validation loss from `validation_step`
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.1,
                                                               verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",  # Name of the logged metric to monitor.
                "interval": "epoch",  # scheduler.step() is called after each epoch
                "frequency": 1,  # scheduler.step() is called once every "frequency" times.
                "strict": True,
            }
        }

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        predictions = self.forward(images)

        # Metrics
        loss = self._loss(predictions, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        predictions = self.forward(images)

        # Compute and log metrics
        self.log("val_loss", self._loss(predictions, labels), on_step=True, on_epoch=True,
                 prog_bar=True)
        self.log("val_f1", self.val_f1(predictions.sigmoid(), labels), on_step=False, on_epoch=True)
        if self.config["n_classes"] > 1:
            self.val_f1_per_class.update(predictions.sigmoid(), labels)

    def on_validation_epoch_end(self):
        if self.config["n_classes"] > 1:
            f1_per_class = self.val_f1_per_class.compute()
            self.val_f1_per_class.reset()

            # Log the per-class F1 scores
            for i, f1 in enumerate(f1_per_class):
                self.log(f'f1_class_{i}/val', f1)

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        predictions = self.forward(images)

        # Metrics
        self.log("test_f1", self.test_f1(predictions.sigmoid(), labels), on_step=False, on_epoch=True)
        if self.config["n_classes"] > 1:
            self.test_f1_per_class.update(predictions.softmax(dim=-1), labels)

    def on_test_epoch_end(self):
        if self.config["n_classes"] > 1:
            f1_per_class = self.test_f1.compute()
            self.test_f1_per_class.reset()

            # Log the per-class F1 scores
            for i, f1 in enumerate(f1_per_class):
                self.log(f'f1_class_{i}', f1)

    def _loss(self, preds, targets):
        if self.config["n_classes"] > 1:
            # Multiclass case
            return self.bce(preds, targets)
        else:
            # Binary case
            return self.bce(preds, targets)
