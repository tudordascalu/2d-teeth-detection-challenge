from pytorch_lightning import LightningModule
from torch.nn import BCEWithLogitsLoss
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.model.unet.unet_components import *
from src.utils.dice_score import DiceScore, DiceLoss


class UNet(LightningModule):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.save_hyperparameters()

        # Hyperparameters
        self.config = config

        # Utilities
        self.dice_score = DiceScore()
        self.dice_loss = DiceLoss()
        self.bce_loss = BCEWithLogitsLoss()

        # Encoder
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 1)

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
        x = self.outc(x)

        return x

    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(),
                            lr=float(self.config["learning_rate"]),
                            weight_decay=1e-8,
                            momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      mode="min",
                                      patience=self.config["scheduler_patience"],
                                      verbose=True)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }}

    def training_step(self, batch, batch_idx):
        inputs, masks, _ = batch["image"], batch["mask"], batch["label"]
        predictions = self.forward(inputs)

        # Log metrics
        loss = self._loss(predictions, masks)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_dice_score", self.dice_score(predictions, masks), prog_bar=True, on_step=True, on_epoch=True)

        # Return loss
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, masks, _ = val_batch["image"], val_batch["mask"], val_batch["label"]
        predictions = self.forward(inputs)

        # Log metrics
        loss = self._loss(predictions, masks)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_dice_score", self.dice_score(predictions, masks), prog_bar=True, on_step=True,
                 on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        inputs, masks, _ = test_batch["image"], test_batch["mask"], test_batch["label"]
        predictions = self.forward(inputs)

        # Log metrics
        loss = self._loss(predictions, masks)
        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_dice_score", self.dice_score(predictions, masks), prog_bar=True, on_epoch=True)

    def _loss(self, prediction, target):
        # Use a combination of cross entropy and dice loss
        return (self.bce_loss(prediction, target) + self.dice_loss(prediction, target)) / 2
