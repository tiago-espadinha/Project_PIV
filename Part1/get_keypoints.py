import numpy as np
from random import randrange
import configparser
import cv2
import os

# Load config file
def load_config(file_path, mode):
    config_edit = configparser.ConfigParser()
    config_edit.read(file_path)
    config = config_edit[mode]
    return config_edit, config

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(params['image'], (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(params['window_name'], params['image'])
        params['points'].append((x, y))

# Load images
config_edit, config = load_config('conf_file.cfg', 'DEFAULT')
frames_directory = config['frames_directory']
image_map = config['image_map']

frame_num = 0
image1 = cv2.imread(frames_directory + image_map, cv2.IMREAD_COLOR)
image2 = cv2.imread(frames_directory + 'frame_0030.jpg', cv2.IMREAD_COLOR)

# Initialize lists to store points
points_image1 = []
points_image2 = []

# Create windows and set a callback function
cv2.namedWindow('Image 1')
cv2.namedWindow('Image 2')
cv2.setMouseCallback('Image 1', click_event, {'points': points_image1, 'image': image1, 'window_name': 'Image 1'})
cv2.setMouseCallback('Image 2', click_event, {'points': points_image2, 'image': image2, 'window_name': 'Image 2'})

# Display images and wait for user input
cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ensure that the same number of points have been selected in both images
if len(points_image1) != len(points_image2) or len(points_image1) < 4:
    raise ValueError("Different number of points selected in images or not enough points")

# Convert points to NumPy arrays
points1 = np.array(points_image1, dtype='int32')
points2 = np.array(points_image2, dtype='int32')

config_file = 'conf_file.cfg'

# Edit config file
config['pts_in_map'] = 'Label ' + str(points1.flatten())[1:-1]
config['pts_in_frame'] = str(30) + ' ' + str(points2.flatten())[1:-1]

with open(config_file, 'w') as configfile:
    config_edit.write(configfile)
