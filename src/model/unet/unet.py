import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.dice_score import DiceScore, DiceLoss


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
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW
        diffD = x2.size()[1] - x1.size()[1]
        diffH = x2.size()[2] - x1.size()[2]
        diffW = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(LightningModule):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.save_hyperparameters()

        # Hyperparameters
        self.config = config

        # Utilities
        self.dice_score = DiceScore()
        self.dice_loss = DiceLoss()
        self.ce_loss = CrossEntropyLoss()
        self.bce_loss = BCELoss()

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

        # Last layer
        x = self.outc(x)
        return x

    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(),
                            lr=float(self.config["learning_rate"]),
                            weight_decay=1e-8,
                            momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      mode="max",
                                      patience=self.config["scheduler_patience"],
                                      verbose=True)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }}

    def training_step(self, batch, batch_idx):
        input, mask, label = batch["image"], batch["mask"], batch["label"]
        prediction = self.forward(input)

        # Log metrics
        loss = self._loss(prediction, mask)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_dice_score", self.dice_score(prediction, mask), prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input, mask, label = val_batch["image"], val_batch["mask"], val_batch["label"]
        prediction = self.forward(input)

        # Log metrics
        self.log("val_loss", self._loss(prediction, mask), prog_bar=True,
                 on_step=True, on_epoch=True)
        self.log("val_dice_score", self.dice_score(prediction, mask), prog_bar=True, on_step=True, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        input, mask, label = test_batch["image"], test_batch["mask"], test_batch["label"]
        prediction = self.forward(input)

        # Log metrics
        self.log("test_loss", self._loss(prediction, mask), prog_bar=True,
                 on_step=True, on_epoch=True)
        self.log("test_dice_score", self.dice_score(prediction, mask), prog_bar=True, on_epoch=True)

    def _loss(self, prediction, target):
        # Use a combination of cross entropy and dice loss
        if self.config["n_classes"] == 1:
            return (self.bce_loss(prediction, target) + self.dice_loss(prediction, target)) / 2
        else:
            return (self.ce_loss(prediction, torch.argmax(target, dim=1)) + self.dice_loss(prediction, target)) / 2
