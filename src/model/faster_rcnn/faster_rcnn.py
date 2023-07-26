import os
import pickle

import torch
import torchvision
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights


class FasterRCNN(pl.LightningModule):
    def __init__(self, config):
        super(FasterRCNN, self).__init__()
        self.save_hyperparameters()
        self.config = config
        # Load the pretrained Faster R-CNN model with ResNet-50 backbone
        if self.config["pretrained"]:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                weights_backbone=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
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
        self.log('loss/train', loss, on_step=True, on_epoch=True)
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
        self.log("loss/val", loss, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def on_validation_epoch_end(self):
        map_dict = self.map_metric.compute()
        self.log("ap/val", map_dict["map"], on_epoch=True, sync_dist=True)
        self.log("ap_50/val", map_dict["map_50"], on_epoch=True, sync_dist=True)
        self.log("ap_75/val", map_dict["map_75"], on_epoch=True, sync_dist=True)
        self.log("mar_100/val", map_dict["mar_100"], on_epoch=True, sync_dist=True)
        self.map_metric.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["targets"]
        predictions = self.model(images)
        self.map_metric.update(predictions, targets)

    def on_test_epoch_end(self):
        map_dict = self.map_metric.compute()
        self.log("ap/test", map_dict["map"], on_epoch=True)
        self.log("ap_50/test", map_dict["map_50"], on_epoch=True)
        self.log("ap_75/test", map_dict["map_75"], on_epoch=True)
        self.log("mar_100/test", map_dict["mar_100"], on_epoch=True, sync_dist=True)
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
        optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config["learning_rate"]), momentum=0.9)
        return optimizer
