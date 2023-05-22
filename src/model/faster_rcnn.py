import torch
import torchvision
import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(pl.LightningModule):
    def __init__(self, config):
        super(FasterRCNN, self).__init__()
        self.save_hyperparameters()
        self.config = config
        # Load the pretrained Faster R-CNN model with ResNet-50 backbone
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # Replace the classifier head with a new one to match the number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config["n_teeth"] + 1)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.forward(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # Set model in training mode to get access to losses
        self.model.train()
        loss_dict = self.forward(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.model.eval()
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config["learning_rate"]), momentum=0.9)
        return optimizer
