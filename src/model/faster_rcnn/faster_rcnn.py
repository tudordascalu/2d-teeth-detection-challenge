import os
import pickle

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
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
        # Average precision
        self.map_metric = MeanAveragePrecision(box_format="xyxy")

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["targets"]
        loss_dict = self.forward(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["targets"]
        # Set model in training mode to get access to losses
        self.model.train()
        loss_dict = self.forward(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        # Set model in eval mode to obtain predictions
        self.model.eval()
        predictions = self.model(images)
        self.map_metric.update(predictions, targets)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        map_dict = self.map_metric.compute()
        self.log("val_ap", map_dict["map"], on_epoch=True)
        self.log("val_ap_50", map_dict["map_50"], on_epoch=True)
        self.log("val_ap_75", map_dict["map_75"], on_epoch=True)
        self.map_metric.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["targets"]
        predictions = self.model(images)
        self.map_metric.update(predictions, targets)

    def test_epoch_end(self, outputs):
        map_dict = self.map_metric.compute()
        self.log("test_ap", map_dict["map"], on_epoch=True)
        self.log("test_ap_50", map_dict["map_50"], on_epoch=True)
        self.log("test_ap_75", map_dict["map_75"], on_epoch=True)
        self.map_metric.reset()

    def on_predict_start(self):
        if not os.path.exists(f"{self.config['output_path']}"):
            os.mkdir(f"{self.config['output_path']}")

    def predict_step(self, batch, batch_idx):
        images, targets, ids = batch["image"], batch["targets"], batch["id"]
        predictions = self.model(images)
        for image, target, id, prediction in zip(images, targets, ids, predictions):
            if not os.path.exists(f"{self.config['output_path']}/{id}"):
                os.mkdir(f"{self.config['output_path']}/{id}")
            # Convert to cpu tensors
            for key, value in target.items():
                if isinstance(value, torch.Tensor):
                    target[key] = value.detach().cpu().numpy()
            for key, value in prediction.items():
                if isinstance(value, torch.Tensor):
                    prediction[key] = value.detach().cpu().numpy()
            # Save dictionaries
            with open(f"{self.config['output_path']}/{id}/targets.npy", "wb") as file:
                pickle.dump(target, file)
            with open(f"{self.config['output_path']}/{id}/predictions.npy", "wb") as file:
                pickle.dump(prediction, file)

    def configure_optimizers(self):
        # TODO: replace with rmsprop
        optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config["learning_rate"]), momentum=0.9)
        return optimizer