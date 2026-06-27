# Laser-Prompted Real-Time Object Segmentation on Smartphones via Cloud Computing

Official implementation of [the paper](https://doi.org/10.1007/s11554-026-01887-z): Laser-prompted real-time object segmentation on smartphones via cloud computing.

![demo](assets/example.gif)

#### Repository contains 3 parts.
- lprtos-cloud - for running the main pipeline on the server, using original PyTorch pt and transformed TensorRT engine (on selected GPU)

 - lprtos-jetson-orin-nano - for running main pipeline on the Jetson Orin Nano, using original PyTorch pt and transformed Tensor

 - lprtos_client - android application

## Test of the Pipeline

For complete evaluation of the system, an additional dataset has been created. [dataset](https://huggingface.co/datasets/striksio/Green-Laser-Guided-Segmentation-Dataset)

Dataset contains 2209 images across different difficult environments. Dataset has both laser box boundaries and masks for the objects the laser points at. COCO format annotations.

lprtos-cloud and lprtos-jetson-orin-nano both contain benchmark scripts.

#### Note: For laser detection, a prediction counts as correct when the top-confidence predicted box center falls inside the GT laser box (center-based, not box overlap). This benchmark method is not used in the original paper, but after evaluation on custom dataset, using IoU box overlap ignores true predictions, where center of that box appears in the laser spot. Thus, if the predicted center is inside gt box, it is considered a true prediction.

## Cloud GPU

### L4 GPU Pytorch pt

#### Laser Detection

| Model | P | R | TP | FP | FN | CenterMed |
|-------|------|------|------|-----|-----|-----------|
| YOLOv5n-LPD.pt | 0.898 | 0.748 | 1639 | 187 | 551 | 1.16px |

#### Object Segmentation

| Model | IoU | Dice | AP50 | AP50:95 |
|-------|-------|-------|-------|---------|
| FastSAM-s.pt | 0.635 | 0.696 | 0.593 | 0.392 |

#### Total Inference Time (median per frame)

| Detection | Segmentation | Inpainting | Total |
|-----------|--------------|------------|-------|
| YOLOv5n-LPD.pt | FastSAM-s.pt | Telea | - |
| 6ms | 13ms | 3ms | 21ms |

---


### L4 GPU TensorRT engine

#### Laser Detection

| Model | P | R | TP | FP | FN | CenterMed |
|-------|------|------|------|-----|-----|-----------|
| YOLOv5n-LPD.engine | 0.893 | 0.747 | 1636 | 196 | 554 | 1.16px |

#### Object Segmentation

| Model | IoU | Dice | AP50 | AP50:95 |
|-------|-------|-------|-------|---------|
| FastSAM-s.engine | 0.622 | 0.682 | 0.573 | 0.376 |

#### Total Inference Time (median per frame)

| Detection | Segmentation | Inpainting | Total |
|-----------|--------------|------------|-------|
| YOLOv5n-LPD.engine | FastSAM-s.engine | Telea | - |
| 4ms | 7ms | 4ms | 16ms |


## Nvidia Jetson Orin Nano GPU

### Jetson Orin Nano Pytorch pt

#### Laser Detection

| Model | P | R | TP | FP | FN | CenterMed |
|-------|------|------|------|-----|-----|-----------|
| YOLOv5n-LPD.pt | 0.898 | 0.749 | 1640 | 187 | 550 | 1.16px |

#### Object Segmentation

| Model | IoU | Dice | AP50 | AP50:95 |
|-------|-------|-------|-------|---------|
| FastSAM-s.pt | 0.635 | 0.696 | 0.593 | 0.392 |

#### Total Inference Time (median per frame)

| Detection | Segmentation | Inpainting | Total |
|-----------|--------------|------------|-------|
| YOLOv5n-LPD.pt | FastSAM-s.pt | Telea | - |
| 14ms | 43ms | 3ms | 60ms |

---

### Jetson Orin Nano TensorRT engine

#### Laser Detection

| Model | P | R | TP | FP | FN | CenterMed |
|-------|------|------|------|-----|-----|-----------|
| YOLOv5n-LPD.engine | 0.893 | 0.748 | 1638 | 196 | 552 | 1.10px |

#### Object Segmentation

| Model | IoU | Dice | AP50 | AP50:95 |
|-------|-------|-------|-------|---------|
| FastSAM-s.engine | 0.622 | 0.682 | 0.573 | 0.377 |

#### Total Inference Time (median per frame)

| Detection | Segmentation | Inpainting | Total |
|-----------|--------------|------------|-------|
| YOLOv5n-LPD.engine | FastSAM-s.engine | Telea | - |
| 9ms | 28ms | 4ms | 42ms |

---

#### Thank you for looking into this repository. This is my first one, and hope not the last one. 
