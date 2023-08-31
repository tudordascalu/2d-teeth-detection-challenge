# Scope

This project aims to build an object detection model that identifies teeth with abnormalities.

## Dataset description

The dataset consists of 2D panoramic X-rays: https://zenodo.org/record/7812323#.ZDQE1uxBwUG.

## Our model

We proposed a multi-step framework that consists of: detection
of dental instances, filtering of healthy instances, and classification of abnormal
instances.

1. Detection of dental instances: We apply Faster-RCNN to identify all teeth in the panoramic X-ray.
2. Filtering of healthy instances: We integrate the encoding path from a pretrained U-net for dental lesion detection
   into the Vgg16 architecture for binary classification of cropped teeth.
3. Classification of abnormal instances: We use the same architecture as in 2 to classify the abnormal teeth.