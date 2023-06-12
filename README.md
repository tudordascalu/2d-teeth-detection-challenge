# Scope

This project aims to build an object detection model that identifies teeth with problems.

# Issues

## Enumeration

- contrast: train_107, train_165, train_236, train_220
- appliance (solvable with inter-teeth dependencies): train_397, train_107, train_564, train_92, train_222, train_404,
  train_208, train_389, train_84
- missing teeth: train_299, train_206, train_523, train_580
- noise: train_576
- wisdom: train_157, train_307, train_484, train_328, train_541
- other: train_607, train_70

# TODOs

## Instance identifier

Implement an instance identifier following the 3d solution submitted for MICCAI:

- measure inter-tooth statistics
- create sample-wise feature matrix using statistics
- design and train network that outputs instance-label pairs
- evaluate improvement compared to FasterRCNN

## Augmentation: in-paiting for authentic tooth removal

Implement solution for generating more samples with appliances, carious lesions, missing teeth by training a GAN to
in-paint teeth.

- identify affected teeth
- train GAN
- use data for tuning model

## Augmentation: affine transformations during training