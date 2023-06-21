import torch.nn
import torchmetrics
from torch.nn import CrossEntropyLoss, Conv2d, Linear
from torchvision import models
import pytorch_lightning as pl


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
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=config["n_classes"])
        self.val_precision = torchmetrics.Precision("multiclass", average="macro", num_classes=config["n_classes"])
        self.val_recall = torchmetrics.Recall("multiclass", average="macro", num_classes=config["n_classes"])
        self.val_f1 = torchmetrics.F1Score("multiclass", average="macro", num_classes=config["n_classes"])
        self.test_acc = torchmetrics.Accuracy("multiclass", num_classes=config["n_classes"])
        self.test_precision = torchmetrics.Precision("multiclass", average="macro", num_classes=config["n_classes"])
        self.test_recall = torchmetrics.Recall("multiclass", average="macro", num_classes=config["n_classes"])
        self.test_f1 = torchmetrics.F1Score("multiclass", average="macro", num_classes=config["n_classes"])

        # Model
        resnet_types = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        self.model = resnet_types[config["resnet_version"]](pretrained=True)
        self.conv1 = Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, bias=False)
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = Linear(linear_size, config["n_classes"])

        # Freeze the early layers (everything but the final layer)
        for name, param in self.model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    def forward(self, input):
        # Convert greyscale to "RGB"
        x = self.conv1(input)

        # Apply Resnet
        x = self.model(x)

        # Apply softmax
        # cls = softmax(x, dim=-1)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config["learning_rate"]), momentum=0.9)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        y_pred = self.forward(images)
        loss = self.ce_loss(y_pred, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        y_pred = self.forward(images)
        y_pred_logits = y_pred.softmax(dim=-1)

        # Compute and log metrics
        self.log("val_loss", self.ce_loss(y_pred, labels), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc(y_pred_logits, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log('val_recall', self.val_recall(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1(y_pred_logits, labels), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        y_pred = self.forward(images)
        y_pred_logits = y_pred.softmax(dim=-1)

        # Compute and log metrics
        self.log("test_loss", self.ce_loss(y_pred, labels), on_step=True, on_epoch=True)
        self.log('test_acc', self.test_acc(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log('test_precision', self.test_precision(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log('test_recall', self.test_recall(y_pred_logits, labels), on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1(y_pred_logits, labels), on_step=False, on_epoch=True)
