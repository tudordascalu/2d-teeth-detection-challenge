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
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=config["n_classes"])
        self.val_precision = torchmetrics.Precision("multiclass", average="macro", num_classes=config["n_classes"])
        self.val_recall = torchmetrics.Recall("multiclass", average="macro", num_classes=config["n_classes"])
        self.val_f1 = torchmetrics.F1Score("multiclass", average="macro", num_classes=config["n_classes"])
        self.test_acc = torchmetrics.Accuracy("multiclass", num_classes=config["n_classes"])
        self.test_precision = torchmetrics.Precision("multiclass", average="macro", num_classes=config["n_classes"])
        self.test_recall = torchmetrics.Recall("multiclass", average="macro", num_classes=config["n_classes"])
        self.test_f1 = torchmetrics.F1Score("multiclass", average="macro", num_classes=config["n_classes"])

        # Model
        vit_types = {
            16: models.vit_b_16,
            32: models.vit_b_32
        }
        self.model = vit_types[config["vit_version"]](weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads.head = torch.nn.Linear(self.model.hidden_dim, config["n_classes"])
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Convert to "RGB"
        x = self.conv1(x)

        # Apply vision transformer
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=float(self.config["learning_rate"]))

        # Reduce learning rate when a metric has stopped improving.
        # "val_loss" is the logged validation loss from `validation_step`
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.1,
                                                               verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val_epoch",  # Name of the logged metric to monitor.
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

        # Compute and log metrics
        self.log("loss/val", self.ce_loss(y_pred, labels), on_step=True, on_epoch=True, prog_bar=True)
        self.log("acc/val", self.val_acc(y_pred_logits, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log("precision/val", self.val_precision(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log("recall/val", self.val_recall(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log("f1/val", self.val_f1(y_pred_logits, labels), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        y_pred = self.forward(images)
        y_pred_logits = y_pred.softmax(dim=-1)

        # Compute and log metrics
        self.log("loss/test", self.ce_loss(y_pred, labels), on_step=True, on_epoch=True)
        self.log("acc/test", self.test_acc(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log("precision/test", self.test_precision(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log("recall/test", self.test_recall(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log("f1/test", self.test_f1(y_pred_logits, labels), on_step=False, on_epoch=True)
