import torch
from torch import nn
import torch.nn.functional as tf
from pytorch_lightning import LightningModule
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import sigmoid, softmax


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        x1 = tf.pad(x1, [diffW // 2, diffW - diffW // 2,
                         diffH // 2, diffH - diffH // 2,
                         diffD // 2, diffD - diffD // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(LightningModule):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.save_hyperparameters()

        # Hyperparameters
        self.config = config

        # Model
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // 2)
        self.up1 = Up(1024, 512 // 2)
        self.up2 = Up(512, 256 // 2)
        self.up3 = Up(256, 128 // 2)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, self.config["n_classes"])

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Last activation
        x = self.outc(x)
        if self.config["n_classes"] == 1:
            x = sigmoid(x)
        else:
            x = softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(),
                            lr=float(self.config["learning_rate"]),
                            weight_decay=1e-8,
                            momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=self.config["scheduler_patience"],
                                      verbose=True)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        accuracy = self._accuracy(y_pred, y)
        self.log("train_loss", loss)
        return {"loss": loss, "train_loss": loss, "train_accuracy": accuracy}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["train_accuracy"] for x in outputs]).mean()
        self.log("step", self.trainer.current_epoch)
        self.log("avg_loss", {"train": avg_loss})
        self.log("avg_accuracy", {"train": avg_accuracy})

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        accuracy = self._accuracy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("step", self.trainer.current_epoch)
        self.log("avg_loss", {"val": avg_loss})
        self.log("avg_accuracy", {"val": avg_accuracy})

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        accuracy = self._accuracy(y_pred, y)
        return {"test_loss": loss, "test_accuracy": accuracy}
