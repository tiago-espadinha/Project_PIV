import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import math
import random
import sys

def parse_config_file(file_path):
    config_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Ignore comments
            if line.startswith('#') or not line:
                continue
            
            # Split the line into tokens
            tokens = line.split()

            # Extract parameter names and values
            param_name = tokens[0]
            param_values = tokens[1:]

            # Check if the token already exists in the dictionary
            if param_name in config_dict:
                # Add new values to the existing token
                config_dict[param_name].extend(param_values)
            else:
                # Create a new entry in the dictionary
                config_dict[param_name] = param_values
    return config_dict


def parse_transforms(config):
    transform_type = config['transforms'][0]
    transform_scope = config['transforms'][1]
    return transform_type, transform_scope


def parse_matches(config):
    label = config['pts_in_map'][0]
    map_points = np.array(config['pts_in_map'][1:], dtype='int32').reshape((-1, 2))
    frame_number = int(config['pts_in_frame'][0])
    frame_points = np.array(config['pts_in_frame'][1:], dtype='int32').reshape((-1, 2))
    return label, map_points, frame_number, frame_points


def generate_random_integers(n, r):
    return [random.randint(0, r-1) for _ in range(n)]


def find_homography(points1, points2, RANSAC, tol = 3.0):
    """
    Returns -> homography between the two set of matching points
    """
    if RANSAC != None:
        outliers = RANSAC(points1, points2, tol=tol)
        points1 = [points1[i] for i in range(len(points1)) if i not in outliers]
        points2 = [points2[i] for i in range(len(points2)) if i not in outliers]
        
    A = []
    for i in range(len(points1)):
        x, y = points1[i][0], points1[i][1]
        xp, yp = points2[i][0], points2[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    H = Vh[-1,:].reshape(3, 3)
    return H

def RANSAC(points1, points2, P=0.99, p=0.5, n_samples=4, tol = 3.0):
    """
    P - Probability of success \n
    p - Probability of all points being inliers (0.5 worst case or if you don't know) \n
    n_samples - Number of points to be used for the calculation of the homography between the images \n
    Returns -> optimal homography between the two set of matching points
    """
    k = math.ceil(np.log(1 - P) / np.log(1 - p**n_samples))
    for i in range(k):
        random_indices = generate_random_integers(n_samples, len(points1))
        sampled_points1 = [points1[index] for index in random_indices]
        sampled_points2 = [points2[index] for index in random_indices]

        # Build the design matrix
        H = find_homography(sampled_points1, sampled_points2, None)
        H_normalized = H / H[2, 2]

        points1 = np.hstack((points1, np.ones((len(points1), 1))))
        points1_transformed = np.copy(points1)

        for j in range(len(points1)):
            points1_transformed[j] = np.dot(H_normalized, points1[j])
            points1_transformed[j] /= points1_transformed[j][2]

        points1_transformed = points1_transformed[:, :2]
        outliers = []
        for j in range(len(points1)):
            # if the distance of the matching points is lower than a certain threshold after projecting a 
            # point from a frame to another than they are considered as inliers
            if np.mean((points2[j] - points1_transformed[j])**2) > tol:
                outliers.append(j)
            
        if i==0:
            min_outliers = outliers
            
        else:
            if len(outliers) < len(min_outliers):
                min_outliers = outliers
        points1 = points1[:, :2]
    
    return min_outliers

def match_and_compute(features, frame_1, frame_2, width, height, H_matrix):
    keypoints1 = features[frame_1][0]
    descriptors1 = features[frame_1][1]
    keypoints2 = features[frame_2][0]
    descriptors2 = features[frame_2][1]
    
    if isinstance(H_matrix[frame_1][frame_2], np.ndarray):
        return H_matrix[frame_1][frame_2], H_matrix
            
    FLANN_INDEX = 1
    index_parameters = dict(algorithm=FLANN_INDEX, trees=5)
    search_parameters = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
            
    # print(f"Number of good matches between frames {frame_2} and {frame_1}: {len(good_matches)}")
    
    if len(good_matches) < 20 and np.abs(frame_1 - frame_2) != 1:
        if isinstance(H_matrix[int((frame_2+frame_1)/2)][frame_2], np.ndarray):
            H1 = np.array(H_matrix[int((frame_2+frame_1)/2)][frame_2])
        else:    
            H1, H_matrix = match_and_compute(features, int((frame_2+frame_1)/2), frame_2, width, height, H_matrix)
        if isinstance(H_matrix[frame_1][int((frame_2+frame_1)/2)], np.ndarray):
            H2 = np.array(H_matrix[frame_1][int((frame_2+frame_1)/2)])
        else:   
            H2, H_matrix = match_and_compute(features, frame_1, int((frame_2+frame_1)/2), width, height, H_matrix)
        H = np.dot(H1, H2)
        H_matrix[frame_1][frame_2] = np.copy(H)
        H_matrix[frame_2][frame_1] = np.copy(np.linalg.inv(H))
        return H, H_matrix

    points_1 = np.float32([keypoints1[m.queryIdx] for m in good_matches]).reshape(-1, 2)
    points_2 = np.float32([keypoints2[m.trainIdx] for m in good_matches]).reshape(-1, 2)
    
    H = find_homography(points_1, points_2, RANSAC)
    H_matrix[frame_1][frame_2] = np.copy(H)
    H_matrix[frame_2][frame_1] = np.copy(np.linalg.inv(H))
    return H, H_matrix
    
def main():

    # Opens the config file
    if len(sys.argv) != 2:
        config_file = 'conf_file.cfg'
    else:
        config_file = sys.argv[1]

    # Get data from the configuration file
    config = parse_config_file(config_file)

    # Check input format
    f=sio.loadmat(config['keypoints_out'][0])
    feat = f['features']
    # print("Feature")
    # print(" -> type: ", type(feat))
    # print(" -> shape: ", feat.shape)
    # print("Keypoint")
    # print(" -> shape: ", feat[0].shape)
    
    transform_type, transform_scope = parse_transforms(config)
    image_map = cv2.imread('map.jpg', cv2.IMREAD_COLOR)
    height, width = image_map.shape[:2]
    _, map_points, keyframe_number, keyframe_points = parse_matches(config)
    
    if transform_type == 'homography':
        H_map = find_homography(keyframe_points, map_points, None)
        H_matrix = np.empty((len(feat), len(feat)), dtype=np.ndarray)
        H_frame_to_map = np.empty(len(feat), dtype=np.ndarray)
        
        H_output = []
        if transform_scope == 'map':    # Compute all homographies between each frame and the map
            for i in range(len(feat)):
                H, H_matrix = match_and_compute(feat, i, keyframe_number, width, height, H_matrix)
                H_final = H_map @ H
                H_lines = [i+1, 0] + H_final.flatten().tolist()  # Include i, and flattened h elements
                H_output.append(H_lines)
                     
        elif transform_scope == 'all':  # Compute all homographies between all frames and the map
            for i in range(len(feat)):
                H, H_matrix = match_and_compute(feat, i, keyframe_number, width, height, H_matrix)
                H_final = H_map @ H
                H_lines = [i+1, 0] + H_final.flatten().tolist()  # Include i, and flattened h elements
                H_output.append(H_lines)
                for j in range(len(feat)):
                    H, H_matrix = match_and_compute(feat, i, j, width, height, H_matrix)
                    H_lines = [i+1, j+1] + H.flatten().tolist()  # Include i, j, and flattened h elements
                    H_output.append(H_lines)
        
        # Matlab output
        mdic = {"H_matrix": H_output}
        sio.savemat(config['transforms_out'][0], mdic)

        # Check image transformation
        # Uncomment the following lines to check the transformation of a frame to the map
        # for i in range(len(feat)):
        #     H_frame_to_map[i] = np.dot(H_map, H_matrix[i][keyframe_number])
        #     frame_filename = f'{frames_directory}frame_{i:04d}.jpg'
        #     image = cv2.imread(frame_filename, cv2.IMREAD_COLOR)
        #     warped_image_to_map = cv2.warpPerspective(image, H_frame_to_map[i], (width, height))
                
        #     fig, axs = plt.subplots(2, figsize=(10, 10))
                
        #     axs[0].imshow(cv2.cvtColor(image_map, cv2.COLOR_BGR2RGB))
        #     axs[0].set_title('Map Image')
            
        #     axs[1].imshow(cv2.cvtColor(warped_image_to_map, cv2.COLOR_BGR2RGB))
        #     axs[1].set_title(f'Warped frame {i} to map')
            
        #     plt.tight_layout()
        #     plt.show()
        
                
if __name__ == '__main__':
    main()
    