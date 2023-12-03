import numpy as np
from random import randrange
import cv2

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(params['image'], (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(params['window_name'], params['image'])
        params['points'].append((x, y))

# Load images
image1 = cv2.imread('processed_videos/trymefirst/frames/frame_0000.jpg', cv2.IMREAD_COLOR)
image2 = cv2.imread('processed_videos/trymefirst/frames/frame_1355.jpg', cv2.IMREAD_COLOR)

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

with open('keypoints.txt', mode='w') as fd:
    fd.write("Points 1: \n" + str(points1))
    fd.write("\nPoints 2: \n" + str(points2))
# fd.close
