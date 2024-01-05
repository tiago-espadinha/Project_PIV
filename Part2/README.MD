# Readme for Image Processing and Vision Project

This repository contains code for processing videos, extracting keypoints, and computing homographies between frames.

## Requirements

open3d
onnxruntime
imread_from_url

## Running the Code

### 1. Process Video (`camera_calibration.py`)

Run the `process_video.py` script to process the video. The configuration file (`config_file.cfg`) includes  the video file path.

```bash
python process_video.py
```

### 2. Get Keypoints (`compute_transform_NN.py`)

Run the `get_keypoint.py` script to manually choose at least 4 keypoints in each image. One of them is frame 0, and the others can be selected interactively. The script will create a `keypoints.txt` with the new coordinates that must be correctly copied to the `config_file.cfg`.

```bash
python get_keypoint.py
```

### 3. Compute Homography (`compute_transform_STEREO.py`)

Run the `compute_homography.py` script to compute homographies using the information from the video processing and manually selected keypoints. The script loads the video and keypoint coordinates from the `config_file.cfg` and uses the `transforms` parameter to operate in `all` or `map` mode.

```bash
python compute_homography.py
```

## Configuration File

The `config_file.cfg` is a configuration file used to specify parameters for video processing, keypoint selection, and homography computation. Ensure that the parameters are correctly set according to your requirements.


from scipy.io import savemat, loadmat
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import configparser
import os
import time
from dataclasses import dataclass
import numpy as np
import onnxruntime
import glob