import torch.nn
import torchmetrics
from torch.nn import CrossEntropyLoss
from torchvision import models
import pytorch_lightning as pl
from torchvision.models import ViT_B_16_Weights


class VisionTransformer(pl.LightningModule):
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
        self.val_f1 = torchmetrics.F1Score("multiclass", average="macro", num_classes=config["n_classes"],
                                           ignore_index=4)
        self.val_f1_per_class = torchmetrics.F1Score("multiclass", average=None, num_classes=config["n_classes"])
        self.test_f1 = torchmetrics.F1Score("multiclass", average="macro", num_classes=config["n_classes"],
                                            ignore_index=4)
        self.test_f1_per_class = torchmetrics.F1Score("multiclass", average=None, num_classes=config["n_classes"])

        # Model
        vit_types = {
            16: models.vit_b_16,
            32: models.vit_b_32
        }
        self.model = vit_types[config["vit_version"]](weights=ViT_B_16_Weights.DEFAULT)
        self.model.heads.head = torch.nn.Linear(self.model.hidden_dim, config["n_classes"])

    def forward(self, x):
        # Apply vision transformer
        x = self.model(x)
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
        y_pred = self.forward(images)
        loss = self.ce_loss(y_pred, labels)
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        y_pred = self.forward(images)
        y_pred_logits = y_pred.softmax(dim=-1)

        # Metrics
        self.val_f1_per_class.update(y_pred_logits, labels)
        self.log("loss/val", self.ce_loss(y_pred, labels), on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1(y_pred_logits, labels), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        f1_per_class = self.val_f1_per_class.compute()
        self.val_f1_per_class.reset()

        # Log the per-class F1 scores
        for i, f1 in enumerate(f1_per_class):
            self.log(f'f1_class_{i}/val', f1)

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        y_pred = self.forward(images)
        y_pred_logits = y_pred.softmax(dim=-1)

        # Metrics
        self.test_f1.update(y_pred_logits, labels)

    def on_test_epoch_end(self):
        f1_per_class = self.test_f1.compute()
        self.test_f1.reset()

        # Log the per-class F1 scores
        for i, f1 in enumerate(f1_per_class):
            self.log(f'f1_class_{i}', f1)
