import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from random import randrange
from PIL import Image
import plotly.graph_objects as go
import plotly.offline as pyo
import cv2

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(params['image'], (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(params['window_name'], params['image'])
        params['points'].append((x, y))

def compute_homography(points1, points2):
    A = []
    for i in range(len(points1)):
        x, y = points1[i][0], points1[i][1]
        xp, yp = points2[i][0], points2[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H

# Load images
image1 = cv2.imread('image_examples\parede1.jpg', cv2.IMREAD_COLOR)
image2 = cv2.imread('image_examples\parede2.jpg', cv2.IMREAD_COLOR)

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
points1 = np.array(points_image1, dtype='float32')
points2 = np.array(points_image2, dtype='float32')

fd = open('keypoints.txt', mode='w')
fd.write("Points 1: \n", str(points1))
fd.write("\nPoints 2: \n", str(points2))
fd.close