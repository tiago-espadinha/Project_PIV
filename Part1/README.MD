# Readme for Part 1 of Image Processing and Vision Project

This repository contains code for processing videos, extracting keypoints, and computing homographies between frames.

## Requirements

The code related to Part 1 of the project uses the following packages:

* configparser
* math
* matplotlib
* numpy
* opencv
* os
* random
* scipy
* sys

## Running the Code

### 1. Process Video (`process_video.py`)

Make sure the following elements of the `config_file.cfg` are adequately set:

* videos (path to the video file)
* frames_directory (directory where the frames of the video will be saved)
* image_map (name of the map image)
* undersampling_factor (factor by which the frames will be undersampled)
* keypoints_out (name of the file where the SIFT features will be saved)

Run the `process_video.py` script to process the video. 

```bash
python process_video.py
```

### 2. Get Keypoints (`get_keypoint.py`)

**Only run if you need to update keypoints on config file**

Make sure the following elements of the `config_file.cfg` are adequately set:

* frames_directory (directory of the frames generated in the previous step)
* image_map (name of the map image)

Run the `get_keypoint.py` script to manually choose at least 4 keypoints in each image. One of them is frame 0, and the others can be selected in the code.

```bash
python get_keypoint.py
```

### 3. Compute Transform (`compute_transform.py`)

Make sure the following elements of the `config_file.cfg` are adequately set:

* frames_directory (directory of the frames generated in the first step)
* image_map (name of the map image generated in the first step)
* transforms (mode of operation: `all` or `map`)
* pts_in_map (map keypoints generated in the previous step)
* pts_in_frame (frame keypoints generated in the previous step)
* keypoints_out (file with SIFT features generated in the first step)
* transforms_out (file where the homographies will be saved)

Run the `compute_homography.py` script to compute homographies using the information from the video processing and manually selected keypoints. The two modes of operation set in the config file are:
* `all`: computes homographies between all frames and saves them in a file
* `map`: computes homographies between the map and all frames and saves them in a file

```bash
python compute_homography.py
```