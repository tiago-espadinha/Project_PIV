import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
import math
import random
import configparser

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def generate_random_integers(n, r):
    return [random.randint(0, r-1) for _ in range(n)]

def compute_homography(points1, points2):
    A = []
    for i in range(len(points1)):
        x, y = points1[i][0], points1[i][1]
        xp, yp = points2[i][0], points2[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H

def apply_homography(H, image):
    h, w = image.shape[:2]
    print(image.shape)
    y_indices, x_indices = np.indices((h, w))
    indices = np.stack((x_indices.ravel(), y_indices.ravel(), np.ones_like(x_indices).ravel())).T
    print("Indices: \n", indices, indices.shape)
    transformed_indices = np.copy(np.array(indices, dtype='float64'))
    for i in range(len(indices)):
        transformed_indices[i] = H @ indices[i]
        transformed_indices[i] /= transformed_indices[i][2]


def find_best_homography(points1, points2, P=0.99, p=0.5, n_samples=4):
    # P = 0.99 # Probability of success
    # p = 0.5 # Probability of all points being inliers (0.5 worst case or if you don't know)
    # n_samples = 4 # Number of points to be used for the calculation of the homography between the images
    k = math.ceil(np.log(1 - P) / np.log(1 - p**n_samples))

    for i in range(k):
        random_indices = generate_random_integers(n_samples, len(points1))
        sampled_points1 = [points1[index] for index in random_indices]
        sampled_points2 = [points2[index] for index in random_indices]

        # Build the design matrix
        H = compute_homography(sampled_points1, sampled_points2)
        H_normalized = H / H[2, 2]
        # H2 = compute_homography(sampled_points2, sampled_points1)
        # H2_normalized = H2 / H2[2, 2]

        points1 = np.hstack((points1, np.ones((len(points1), 1))))
        points1_transformed = np.copy(points1)
        # points2 = np.hstack((points2, np.ones((len(points2), 1))))
        # points2_transformed = np.copy(points2)

        for j in range(len(points1)):
            points1_transformed[j] = H_normalized @ points1[j]
            points1_transformed[j] /= points1_transformed[j][2]

        # for j in range(len(points2)):
        #     points2_transformed[j] = H2_normalized @ points2[j]
        #     points2_transformed[j] /= points2_transformed[j][2]

        points1_transformed = points1_transformed[:, :2]

        if i==0:
            best_H = H_normalized
            min_error = np.mean((points2 - points1_transformed)**2)
        else:
            error = np.mean((points2 - points1_transformed)**2)
            if error < min_error:
                best_H = H_normalized
                min_error = error
        points1 = points1[:, :2]
        # points2 = points2[:, :2]

    return best_H
    
def main():

    # what should be outputed
    f=scipy.io.loadmat('Project_PIV/specs/surf_features.mat')
    feat = f['features']
    print("Feature")
    print(" -> type: ", type(feat))
    print(" -> shape: ", feat.shape)
    print("Keypoint")
    print(" -> shape: ", feat[0,2].shape)

    #Example (will be changed)
    image1 = cv2.imread('image_examples\parede1.jpg', cv2.IMREAD_COLOR)
    image2 = cv2.imread('image_examples\parede2.jpg', cv2.IMREAD_COLOR)
    
    sift = cv2.SIFT.create()
    
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    
    # Create a Brute Force Matcher
    bf = cv2.BFMatcher()
    
    # Match descriptors using KNN (k-nearest neighbors)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
    
    # Get corresponding points
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # # Draw the matches
    # img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('Matches', img_matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    points1 = points1.reshape((points1.shape[0], points1.shape[2]))
    # print("Points 1: \n", points1)
    points2 = points2.reshape((points2.shape[0], points2.shape[2]))
    # print("Points 2: \n", points2)
    
    P = 0.99 # Probability of success
    p = 0.5 # Probability of all points being inliers (0.5 worst case or if you don't know)
    n_samples = 4 # Number of points to be used for the calculation of the homography between the images
    
    H = find_best_homography(points1, points2, P=0.99, p=0.5, n_samples=4)
    height, width = image2.shape[:2]
    warped_image_1 = cv2.warpPerspective(image1, H, (width, height))
    
    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, figsize=(10, 10))
    
    # Original left image
    axs[0].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image 2')
    
    # Original right image
    axs[1].imshow(cv2.cvtColor(warped_image_1, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Warped Image 1')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()
    
