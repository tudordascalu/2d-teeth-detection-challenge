import os

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes


class FasterRCNN(pl.LightningModule):
    def __init__(self, config):
        super(FasterRCNN, self).__init__()
        self.save_hyperparameters()
        self.config = config

        # Model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=self.config["pretrained"],
                                                                          pretrained_backbone=self.config["pretrained"])
        # Replace the classifier head with a new one to match the number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

        # Utilities
        self.map_metric = MeanAveragePrecision(box_format="xyxy")

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch["image"], batch["targets"]
        loss_dict = self.forward(inputs, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch["image"], batch["targets"]

        # Set model in training mode to get access to losses
        self.model.train()
        loss_dict = self.forward(inputs, targets)
        loss = sum(loss for loss in loss_dict.values())

        # Set model in eval mode to obtain predictions
        self.model.eval()
        predictions = self.model(inputs)

        # Save predictions
        os.makedirs(f"output/faster_rcnn/version_{self.logger.version}/epoch_{self.current_epoch}", exist_ok=True)
        colors = np.array(["black", "red", "green", "blue", "yellow", "orange"])
        for i, (input, prediction, target) in enumerate(zip(inputs, predictions, targets)):
            input_int = (input * 255).type(torch.uint8)
            output_image = draw_bounding_boxes(image=input_int,
                                               boxes=prediction["boxes"],
                                               colors=colors[prediction["labels"].detach().cpu()].tolist(),
                                               width=5)
            target_image = draw_bounding_boxes(image=input_int,
                                               boxes=target["boxes"],
                                               colors=colors[target["labels"].cpu()].tolist(),
                                               width=5)
            # Figure
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(output_image.permute(1, 2, 0).detach().cpu())
            ax[1].imshow(target_image.permute(1, 2, 0).detach().cpu())
            ax[0].set_title("Output image")
            ax[1].set_title("Target image")
            plt.savefig(
                f"output/faster_rcnn/version_{self.logger.version}/epoch_{self.current_epoch}/{batch_idx}_{i}.png")
            plt.close("all")

        # Metrics
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

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config["learning_rate"]), momentum=0.9)
        return optimizer
