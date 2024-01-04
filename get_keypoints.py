import numpy as np
from random import randrange
import cv2
import configparser
import os

# Create a new folder for the frames
def get_frame_path(file_path):
    file_dir, filename = os.path.split(file_path)
    dir_name, _ = os.path.splitext(filename)
    new_file_dir = os.path.join(file_dir, dir_name) + '/'
    return new_file_dir


# Load config file
def load_config(file_path, mode):
    config_edit = configparser.ConfigParser()
    config_edit.read(file_path)
    config = config_edit[mode]
    return config_edit, config


# Get user input
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(params['image'], (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(params['window_name'], params['image'])
        params['points'].append((x, y))


# Load specific frame
def load_frame(vid, frame_num):
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = vid.read()
    if not ret:
        print(f"Error reading frame {frame_num}")
        return
    return frame

""" Tesla dataset:
video_directory = 'Tesla/specs/'
video_filename = 'backcamera_s1.mp4'
map_directory = 'processed_videos/TeslaVC_carreira/'
map_filename = 'map.jpg'
frames_directory = 'processed_videos/TeslaVC_carreira/back_frames/'
"""
""" Drone dataset:
video_directory = 'videos/'
video_filename = 'trymefirst.mp4'
map_directory = 'processed_videos/trymefirst/frames'
map_filename = 'frame_0045.jpg'
frames_directory = 'processed_videos/trymefirst/frames'
"""

keyframe_num = 0
config_file = 'part1.cfg'
config_edit, config = load_config(config_file, 'DEFAULT')
map_directory = 'processed_videos/trymefirst_lisbon/frames/'
map_filename = 'frame_0045.jpg'
frames_directory = 'processed_videos/trymefirst_lisbon/frames/'
video_directory = 'videos/'
video_filename = 'trymefirst_lisbon.mp4'

#image1 = load_frame(video, 0)
#image2 = load_frame(video, frame_num)
map_image = cv2.imread(map_directory + map_filename, cv2.IMREAD_COLOR)
keyframe_image = cv2.imread(frames_directory + f'frame_{keyframe_num:04d}.jpg', cv2.IMREAD_COLOR)
# Initialize lists to store points
points_image1 = []
points_image2 = []

# Create windows and set a callback function
cv2.namedWindow('Map Image')
cv2.namedWindow(f'Frame {keyframe_num}')
cv2.setMouseCallback('Map Image', click_event, {'points': points_image1, 'image': map_image, 'window_name': 'Map Image'})
cv2.setMouseCallback(f'Frame {keyframe_num}', click_event, {'points': points_image2, 'image': keyframe_image, 'window_name': f'Frame {keyframe_num}'})

# Display images and wait for user input
cv2.imshow('Map Image', map_image)
cv2.imshow(f'Frame {keyframe_num}', keyframe_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ensure that the same number of points have been selected in both images
if len(points_image1) != len(points_image2) or len(points_image1) < 4:
    raise ValueError("Different number of points selected in images or not enough points")

# Convert points to NumPy arrays
points1 = np.array(points_image1, dtype='int32')
points2 = np.array(points_image2, dtype='int32')

# Edit config file
# TODO: Avoid removing comments on config file
config['pts_in_map'] = 'Label ' + str(points1.flatten())[1:-1]
config['pts_in_frame'] = str(keyframe_num) + ' ' + str(points2.flatten())[1:-1]
config['image_map'] = map_directory + map_filename
config['videos'] = video_directory + video_filename

with open(config_file, 'w') as configfile:
    config_edit.write(configfile)

# with open('keypoints.txt', mode='w') as fd:
#     fd.write("Points 1: \n" + str(points1))
#     fd.write("\nPoints 2: \n" + str(points2))