import torch.nn
import torchmetrics
from torch.nn import CrossEntropyLoss, Conv2d, Linear
from torchvision import models
import pytorch_lightning as pl
from torchvision.models import ResNet50_Weights


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
        self.train_f1 = torchmetrics.F1Score("multiclass", average="macro", num_classes=config["n_classes"])
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=config["n_classes"])
        self.val_precision = torchmetrics.Precision("multiclass", average="macro", num_classes=config["n_classes"])
        self.val_recall = torchmetrics.Recall("multiclass", average="macro", num_classes=config["n_classes"])
        self.val_f1 = torchmetrics.F1Score("multiclass", average="macro", num_classes=config["n_classes"],
                                           ignore_index=4)
        self.val_f1_per_class = torchmetrics.F1Score("multiclass", average=None, num_classes=config["n_classes"],
                                                     ignore_index=4)
        self.test_acc = torchmetrics.Accuracy("multiclass", num_classes=config["n_classes"])
        self.test_precision = torchmetrics.Precision("multiclass", average="macro", num_classes=config["n_classes"])
        self.test_recall = torchmetrics.Recall("multiclass", average="macro", num_classes=config["n_classes"])
        self.test_f1 = torchmetrics.F1Score("multiclass", average=None, num_classes=config["n_classes"])

        # Model
        resnet_types = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        self.model = resnet_types[config["resnet_version"]](weights=ResNet50_Weights.DEFAULT)
        self.conv1 = Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, bias=False)
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = Linear(linear_size, config["n_classes"])

        # Freeze the early layers (everything but the final layer)
        for name, param in self.model.named_parameters():
            # if "layer4" not in name and "fc" not in name:
            if "fc" not in name:
                param.requires_grad = False

    def forward(self, input):
        # Convert greyscale to "RGB"
        x = self.conv1(input)

        # Apply Resnet
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

        y_pred_logits = y_pred.softmax(dim=-1)
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1(y_pred_logits, labels), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        y_pred = self.forward(images)
        y_pred_logits = y_pred.softmax(dim=-1)

        # Compute and log metrics
        self.val_f1_per_class.update(y_pred_logits, labels)

        self.log("loss/val", self.ce_loss(y_pred, labels), on_step=True, on_epoch=True, prog_bar=True)
        self.log("acc/val", self.val_acc(y_pred_logits, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log("precision/val", self.val_precision(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log("recall/val", self.val_recall(y_pred_logits, labels), on_step=False, on_epoch=True)
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

        # Update metrics
        self.test_f1.update(y_pred_logits, labels)

        # Log metrics
        self.log("loss/test", self.ce_loss(y_pred, labels), on_step=True, on_epoch=True)
        self.log("acc/test", self.test_acc(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log("precision/test", self.test_precision(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log("recall/test", self.test_recall(y_pred_logits, labels), on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        f1_per_class = self.test_f1.compute()
        self.test_f1.reset()

        # Log the per-class F1 scores
        for i, f1 in enumerate(f1_per_class):
            self.log(f'f1_class_{i}', f1)
