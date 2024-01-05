# Readme for Part 2 of Image Processing and Vision Project

Github repository: https://github.com/tiago-espadinha/Project_PIV

This repository contains code for camera calibration and depth map reconstruction, through neural network or SIFT matching.

## Requirements

The code related to Part 2 of the project uses the following packages:

* configparser
* dataclasses
* glob
* matplotlib
* numpy
* onnxruntime
* open3d
* opencv
* os
* scipy
* time

## Running the Code

### 1. Camera Calibration (`camera_calibration.py`)

Make sure the following elements of the `config_file.cfg` are adequately set:

* calib_path (path to the calibration files)
* dataset_path (path to input images)
* left_image_name (name of the left stereo image)
* right_image_name (name of the right stereo image)

Run the `camera_calibration.py` script to get the camera intrinsics and extrinsics.

```bash
python camera_calibration.py
```

### 2. Compute Depth Map (`compute_transform_NN.py` and `compute_transform_STEREO.py`)

**IMPORTANT: If a neural network is not present go to the associated github page (https://github.com/tiago-espadinha/Project_PIV) and download the one present in the folder `models`**

Make sure the following elements of the `config_file.cfg` are adequately set:

* calib_path (path to the calibration files)
* dataset_path (path to input images)
* left_image_name (name of the left stereo image)
* right_image_name (name of the right stereo image)
* model_path (path to the ONNX neural network model)
* output_path_NN (path to save results)
* output_path_STEREO (path to save results)

Run either the `compute_transform_NN.py` or the `compute_transform_STEREO.py` scripts to compute the depth map. The first will do so using a pre-trained neural network, while the second will do so using SIFT matches between prespectives. **Note that to use the `compute_transform_NN.py` script, `stereo_model.py` also needs to be present**

```bash
python compute_transform_NN.py
```

or

```bash
python compute_transform_STEREO.py
```