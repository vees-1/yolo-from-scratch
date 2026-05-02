<div align="center">

# YOLO From Scratch

**YOLOv1 and YOLOv3 object detection implemented in PyTorch**

[![Repository](https://img.shields.io/badge/GitHub-yolo--from--scratch-black?style=flat-square)](https://github.com/vees-1/yolo-from-scratch)
[![Framework](https://img.shields.io/badge/PyTorch-object%20detection-red?style=flat-square)](#)

</div>

---

## What Is This Project?

This project implements YOLO-style object detection models in PyTorch.

The goal is to understand how object detectors work internally instead of only using a prebuilt library. The notebooks build the model architecture, dataset pipeline, loss functions, decoding logic, non-maximum suppression, training loop, and mAP evaluation.

Two detectors are implemented:

- **YOLOv1** on Pascal VOC.
- **YOLOv3** on Pascal VOC and COCO-style data.

This is a learning and research project, not a production detector package.

---

## Why This Project Exists

YOLO is one of the most important real-time object detection families. A detector must solve three problems at the same time:

- find where objects are,
- classify what each object is,
- remove duplicate detections.

This project was made to learn that full pipeline from the inside:

- how images and bounding boxes are loaded,
- how labels are converted into grid targets,
- how anchor boxes work,
- how detector heads predict boxes/classes/objectness,
- how loss is calculated,
- how predictions are decoded,
- how NMS removes duplicates,
- how mAP@0.5 is measured.

---

## What Has Been Built

### YOLOv1

Implemented in `yolov1.ipynb`.

The YOLOv1 notebook includes:

- Pascal VOC dataset loading,
- 7x7 prediction grid,
- 2 bounding boxes per grid cell,
- 20 Pascal VOC classes,
- GoogLeNet/ImageNet backbone,
- YOLO-style detection head,
- custom YOLOv1 loss,
- prediction decoding,
- non-maximum suppression,
- mAP@0.5 evaluation,
- training and checkpoint saving.

Best result recorded in the notebook:

```text
YOLOv1 on Pascal VOC
Best mAP@0.5: 0.2972
```

### YOLOv3

Implemented in `yolov3.ipynb`.

The YOLOv3 notebook includes:

- Darknet-53-style backbone,
- residual blocks,
- three detection scales,
- anchor-based prediction,
- pretrained Darknet-53 weight loading through `timm`,
- backbone freezing/unfreezing,
- custom YOLOv3 loss,
- decoding at multiple scales,
- class-wise NMS,
- mAP@0.5 evaluation,
- training on Pascal VOC,
- COCO-format dataset support.

Best result recorded in the notebook:

```text
YOLOv3 on Pascal VOC
Best mAP@0.5: 0.6536
```

### Inference / Testing

Implemented in `test.ipynb`.

The test notebook loads saved YOLOv1 and YOLOv3 checkpoints from `models/`, runs detection on an image, decodes predictions, applies NMS, and draws boxes with class labels.

---

## Architecture

```text
Images + annotations
   |
   v
datasets.py
   |
   |-- VOCDataset
   |   reads Pascal VOC images and XML annotations
   |
   |-- COCODataset
   |   reads COCO images with YOLO-format label files
   |
   |-- collate_fn
   |   keeps variable-length bounding boxes per image
   v
Model notebook
   |
   |-- yolov1.ipynb
   |   grid detector with one prediction scale
   |
   |-- yolov3.ipynb
   |   anchor detector with three prediction scales
   v
Training loop
   |
   |-- forward pass
   |-- custom detection loss
   |-- optimizer step
   |-- validation loss
   |-- mAP@0.5 calculation
   v
Saved checkpoint
   |
   v
test.ipynb
   |
   |-- load model
   |-- decode boxes
   |-- apply NMS
   |-- visualize detections
```

---

## Repository Map

```text
yolov1.ipynb
  YOLOv1 model, loss, training, decoding, NMS, mAP evaluation

yolov3.ipynb
  YOLOv3 model, Darknet-style backbone, anchors, multi-scale training, mAP evaluation

test.ipynb
  Loads trained checkpoints and visualizes detections

datasets.py
  Pascal VOC and COCO dataset classes, augmentation, collate function, dataloader helper

requirements.txt
  Python dependencies

data/
  Local VOC/COCO datasets
  Not intended to be committed

models/
  Local trained checkpoints
  Not intended to be committed
```

---

## Dataset Support

### Pascal VOC

Used for YOLOv1 and YOLOv3 training.

Expected structure:

```text
data/VOC/VOC2007/
  JPEGImages/
  Annotations/
  ImageSets/Main/

data/VOC/VOC2012/
  JPEGImages/
  Annotations/
  ImageSets/Main/
```

The notebooks use:

- VOC 2007 train,
- VOC 2012 trainval,
- VOC 2007 test.

### COCO

YOLOv3 includes COCO-style support.

Expected structure:

```text
data/COCO2017/
  train2017/
  val2017/
  labels/train2017/
  labels/val2017/
```

Labels are expected in YOLO text format:

```text
class_id center_x center_y width height
```

All coordinates are normalized to `[0, 1]`.

---

## How Training Works

### YOLOv1

The image is divided into a `7 x 7` grid. Each grid cell predicts:

- object confidence,
- box center,
- box size,
- class probabilities.

Training uses a custom loss that combines:

- coordinate loss,
- objectness loss,
- no-object loss,
- class prediction loss.

### YOLOv3

YOLOv3 predicts at three scales:

- `13 x 13` for large objects,
- `26 x 26` for medium objects,
- `52 x 52` for small objects.

Each scale uses three anchors. The loss trains:

- center offsets,
- width/height anchor adjustment,
- objectness,
- class probabilities.

---

## Results

| Model | Dataset | Best Metric |
| --- | --- | --- |
| YOLOv1 | Pascal VOC | `mAP@0.5 = 0.2972` |
| YOLOv3 | Pascal VOC | `mAP@0.5 = 0.6536` |

The YOLOv3 result is much stronger because it uses a better backbone, anchor boxes, feature pyramids, and multi-scale detection.

---

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks:

```bash
jupyter notebook
```

Open:

```text
yolov1.ipynb
yolov3.ipynb
test.ipynb
```

---

## Current Limitations

- The project is notebook-first, not packaged as a reusable Python library.
- Training requires local VOC/COCO data.
- Checkpoints are local and not committed.
- mAP evaluation is implemented for learning clarity, not optimized for speed.
- Data augmentation is basic.
- No full experiment tracking is included.

---

## Future Plans

- Move model code from notebooks into reusable Python modules.
- Add a clean training CLI.
- Add config files for model, dataset, and training settings.
- Add stronger augmentations such as mosaic, random crop, and color jitter schedules.
- Add checkpoint download links or release artifacts.
- Add sample inference images to the README.
- Add COCO training results once long training is completed.
- Add unit tests for box conversion, IoU, NMS, and target assignment.

---

## Project Status

This project demonstrates the complete object detection pipeline:

```text
image + boxes
  -> dataset loader
  -> YOLO model
  -> custom detection loss
  -> training loop
  -> mAP evaluation
  -> saved checkpoint
  -> inference visualization
```

It is useful as a resume project because it shows understanding of deep learning beyond calling a pretrained detector.
