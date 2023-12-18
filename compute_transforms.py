import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
import math
import random
import configparser

def load_config(file_path, mode):
    config = configparser.ConfigParser()
    config.read(file_path)
    config = config[mode]
    return config

def parse_transforms(config):
    transform_type = config.get('transforms').split()[0]
    transform_scope = config.get('transforms').split()[1]
    return transform_type, transform_scope

def parse_matches(config):
    label = config.get('pts_in_map').split()[0]
    map_points = np.array(config.get('pts_in_map').split()[1:], dtype='int32').reshape((-1, 2))
    frame_number = int(config.get('pts_in_frame').split()[0])
    frame_points = np.array(config.get('pts_in_frame').split()[1:], dtype='int32').reshape((-1, 2))
    return label, map_points, frame_number, frame_points

def generate_random_integers(n, r):
    return [random.randint(0, r-1) for _ in range(n)]

def compute_homography(points1, points2):
    """
    Returns -> homography between the two set of matching points
    """
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

def find_best_homography(points1, points2, P=0.99, p=0.5, n_samples=4):
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
        H = compute_homography(sampled_points1, sampled_points2)
        H_normalized = H / H[2, 2]

        points1 = np.hstack((points1, np.ones((len(points1), 1))))
        points1_transformed = np.copy(points1)

        for j in range(len(points1)):
            points1_transformed[j] = H_normalized @ points1[j]
            points1_transformed[j] /= points1_transformed[j][2]

        points1_transformed = points1_transformed[:, :2]
        inliers = 0
        for j in range(len(points1)):
            # if the distance of the matching points is lower than a certain threshold after projecting a 
            # point from a frame to another than they are considered as inliers
            if np.mean((points2[j] - points1_transformed[j])**2) < 3.0:
                inliers += 1
        error = np.mean((points2 - points1_transformed)**2)
            
        if i==0:
            best_H = H_normalized
            min_error = error
            max_inliers = inliers
        else:
            if inliers > max_inliers:
                best_H = H_normalized
                min_error = error
                max_inliers = inliers
        points1 = points1[:, :2]
    print(f"Min Error: {min_error} with {max_inliers} inliers")
    return best_H, min_error

def match_and_compute(features, frame_1, frame_2, width, height):
    keypoints_1 = features[frame_1][0]
    descriptors_1 = features[frame_1][1]
    keypoints_2 = features[frame_2][0]
    descriptors_2 = features[frame_2][1]
    
    # Create a Brute Force Matcher
    bf = cv2.BFMatcher()
    
    # Match descriptors using KNN (k-nearest neighbors)
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
    
    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
            
    print(f"Number of good matches between frames {frame_2} and {frame_1}: {len(good_matches)}")
    
    if len(good_matches) < 20:
        H1 = match_and_compute(features, int((frame_2+frame_1)/2), frame_2, width, height)
        H2 = match_and_compute(features, frame_1, int((frame_2+frame_1)/2), width, height)
        return H1 @ H2

    points_1 = np.float32([keypoints_1[m.queryIdx] for m in good_matches]).reshape(-1, 2)
    points_2 = np.float32([keypoints_2[m.trainIdx] for m in good_matches]).reshape(-1, 2)
    
    H, error = find_best_homography(points_1, points_2, P=0.99, p=0.5, n_samples=4)
    
    return H
    
def main():
    """ Drone dataset:
    frames_directory = 'processed_videos/trymefirst/frames/'
    """
    """ Tesla dataset:
    frames_directory =  'processed_videos/TeslaVC_carreira/back_frames/'
    """
    frames_directory = 'processed_videos/trymefirst/frames/'
    
    # keypoints1 = feat[frame1_number][0]
    # descriptors1 = feat[frame1_number][1]
    # keypoints2 = feat[frame2_number][0]
    # descriptors2 = feat[frame2_number][1]
    
    # Get data from the configuration file
    config = load_config('part1.cfg', 'DEFAULT')
    # what should be outputed
    f=scipy.io.loadmat(config['keypoints_out'])
    feat = f['features']
    print("Feature")
    print(" -> type: ", type(feat))
    print(" -> shape: ", feat.shape)
    print("Keypoint")
    print(" -> shape: ", feat[0].shape)
    
    transform_type, transform_scope = parse_transforms(config)
    image_map = cv2.imread(config['image_map'], cv2.IMREAD_COLOR)
    height, width = image_map.shape[:2]
    _, map_points, keyframe_number, keyframe_points = parse_matches(config)
    
    if transform_type == 'homography':
        H_map = compute_homography(keyframe_points, map_points)
        image_keyframe = cv2.imread(f'{frames_directory}frame_{keyframe_number:04d}.jpg', cv2.IMREAD_COLOR)
        
        if transform_scope == 'map':    # Compute all homographies between each frame and the map
            for i in range(len(feat)):
                frame_filename = f'{frames_directory}frame_{i:04d}.jpg'
                image = cv2.imread(frame_filename, cv2.IMREAD_COLOR)
                
                H = match_and_compute(feat, i, keyframe_number, width, height)     
                warped_image = cv2.warpPerspective(image, H, (width, height))  
                warped_image_to_map = cv2.warpPerspective(warped_image, H_map, (width, height))
                
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                
                axs[0][0].imshow(cv2.cvtColor(image_map, cv2.COLOR_BGR2RGB))
                axs[0][0].set_title('Map Image')
                
                axs[0][1].imshow(cv2.cvtColor(warped_image_to_map, cv2.COLOR_BGR2RGB))
                axs[0][1].set_title(f'Warped frame {i} to map')
                
                axs[1][0].imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
                axs[1][0].set_title(f'Warped frame {i} to keyframe{keyframe_number}')
                
                axs[1][1].imshow(cv2.cvtColor(image_keyframe, cv2.COLOR_BGR2RGB))
                axs[1][1].set_title(f'Keyframe {keyframe_number}')
                
                plt.tight_layout()
                plt.show() 
                
if __name__ == '__main__':
    main()
    
