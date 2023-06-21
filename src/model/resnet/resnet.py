import pytorch_lightning as pl
import torch.nn
from torch import softmax
from torch.nn import CrossEntropyLoss, Conv2d, Linear
from torchvision import models


class ResNet(pl.LightningModule):
    def __init__(self, config):
        """
        :param transfer: flag that controls transfer learning
        :param tune_fc_only: flag that controls which layers are fine-tuned
        """
        super().__init__()
        self.__dict__.update(locals())

        # Hyperparameters
        self.save_hyperparameters()
        self.config = config

        # Utils
        self.ce_loss = CrossEntropyLoss()

        # Model
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        self.model = resnets[config["resnet_version"]](pretrained=True)
        self.conv1 = Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, bias=False)
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = Linear(linear_size, config["n_classes"])

    def forward(self, input):
        # Convert greyscale to "RGB"
        x = self.conv1(input)

        # Apply Resnet
        x = self.model(x)

        # Apply softmax
        cls = softmax(x, dim=-1)

        return cls

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config["learning_rate"]), momentum=0.9)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        y_pred = self.forward(images)
        loss = self.ce_loss(y_pred, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        y_pred = self.forward(images)
        loss = self.ce_loss(y_pred, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        y_pred = self.forward(images)
        loss = self.ce_loss(y_pred, labels)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
