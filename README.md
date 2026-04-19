# Spacestation Objects Detection - Duality

A YOLOv8-based object detection model trained on synthetic data generated using Duality's FalconEditor simulation software, tested on real-world images.

## Overview

This project demonstrates how synthetic data from a simulated space station environment can be used to train an object detection model that works on real-world images.

## Dataset

The training dataset is hosted on Google Drive. It will be automatically accessed when running the notebook in Google Colab.

## Run Instructions

1. Open `syntheticDataWorks_multiclass.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Run all cells top to bottom

> Make sure to enable GPU: `Edit > Notebook settings > Hardware Accelerator > T4 GPU`

## Notebook Features

### Training
Trains a YOLOv8 model on the synthetic dataset (~20-30 mins). Outputs final metrics in this format:
```
Class     Images  Instances    Box(P       R     mAP50  mAP50-95)
  all        138        136    0.978   0.993     0.993      0.980
```

### Training Metrics & Graphs
After training, the notebook displays `results.png` showing:
- Loss curves (should trend downward)
- Precision & Recall curves (should trend upward)
- mAP50 curve (should trend upward)

### Prediction on Real-World Images
Runs `predict.py` to test the trained model on real-world test images. Outputs annotated prediction images and displays a grid of results directly in the notebook.

### Report Generation
Runs `report/report.py` to generate a final summary report of the model's performance saved to `report/generatedreport.d`.

### Camera Distance Analysis
Runs `camera_distance_analysis.py` which compares the synthetic training data and real-world test images by analyzing the relative bounding box size of detected objects. Outputs:
- Height and width comparison graphs per class
- Recommended camera distance adjustments to improve future synthetic data generation

## Project Structure

```
├── Output/
│   ├── train/labels/
│   ├── val/labels/
│   ├── classes.txt
│   ├── train.py
│   ├── predict.py
│   ├── visualize.py
│   └── yolo_params.yaml
├── testImages/labels/
├── report/report.py
└── syntheticDataWorks_multiclass.ipynb
```
