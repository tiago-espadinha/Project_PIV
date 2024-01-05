import numpy as np
from scipy.io import savemat, loadmat
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import configparser
import os


# Load config file
def load_config(file_path, mode):
    config = configparser.ConfigParser()
    config.read(file_path)
    config = config[mode]
    return config


# Create a new folder if it doesn't exist
def path_exists(file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def compute_dlt_transformation_matrix(points_3d, points_2d):
    """
    Compute the Direct Linear Transformation (DLT) transformation matrix.

    Parameters:
    - points_3d: 3D points in homogeneous coordinates (Nx4)
    - points_2d: Corresponding 2D points (Nx2)

    Returns:
    - T: Transformation matrix (3x4)
    """
    # Ensure homogeneous coordinates
    points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_2d = np.hstack((points_2d, np.ones((points_2d.shape[0], 1))))

    # Construct the matrix A
    A = []
    for i in range(len(points_3d)):
        X = points_3d[i]
        x = points_2d[i]

        A.append([-X[0], -X[1], -X[2], -1, 0, 0, 0, 0, x[0] * X[0], x[0] * X[1], x[0] * X[2], x[0]])
        A.append([0, 0, 0, 0, -X[0], -X[1], -X[2], -1, x[1] * X[0], x[1] * X[1], x[1] * X[2], x[1]])

    A = np.array(A)

    # Perform SVD
    _, _, V = np.linalg.svd(A)

    # Extract the solution (last column of V)
    transformation_matrix = V[-1].reshape((3, 4))

    return transformation_matrix

def compute_rotation_translation(points1, points2, K1, K2):
    """
    Compute the rotation matrix and translation vector between two camera angles.

    Parameters:
    - points1: Corresponding points from camera 1 (Nx2)
    - points2: Corresponding points from camera 2 (Nx2)
    - K1: Intrinsic matrix for camera 1 (3x3)
    - K2: Intrinsic matrix for camera 2 (3x3)

    Returns:
    - R: Rotation matrix (3x3)
    - t: Translation vector (3x1)
    """

    # Compute the essential matrix
    F, mask = cv2.findFundamentalMat(points1, points2, method=cv2.FM_LMEDS)
    F_normalized = F / np.linalg.norm(F)
    E = np.dot(K2.T, np.dot(F, K1))
    U, S, Vt = np.linalg.svd(E)
    S[2] = 0
    S = np.diag(S)
    E_normalized = np.dot(U, np.dot(S, Vt))

    # Recover the rotation and translation from the essential matrix
    _, R, t, _ = cv2.recoverPose(E_normalized, points1, points2, K2)

    return F, mask, E_normalized, R, t

def main():
    
    config = load_config('conf_file.cfg', 'DEFAULT')
    output_path = config['output_path_STEREO']
    path_exists(output_path)

    # Load images
    dataset_path = config['dataset_path']
    img_left_name = config['left_image_name']
    img_right_name = config['right_image_name']

    img_left = cv2.imread(dataset_path + img_left_name, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(dataset_path + img_right_name, cv2.IMREAD_GRAYSCALE)
    frameSize = (600, 800)
    img_left = cv2.resize(img_left, frameSize)
    img_right = cv2.resize(img_right, frameSize)
    
    calib_path = config['calib_path']
    calib_path_out = calib_path + '/output/'
    calib_data_R = loadmat(calib_path_out + "calib_R.mat")
    calib_data_L = loadmat(calib_path_out + "calib_L.mat")
   
    # Detect keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img_left, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img_right, None)
    
    FLANN_INDEX = 1
    index_parameters = dict(algorithm=FLANN_INDEX, trees=5)
    search_parameters = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    points1 = np.int32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.int32([keypoints2[m.trainIdx].pt for m in good_matches])
    
    img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
    img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    
    # Plot Epipolar Lines on Image 1
    image1_with_matches = img_left.copy()
    for point in points1:
        x, y = map(int, [point[0], point[1]])
        cv2.circle(image1_with_matches, (x, y), 5, (0, 255, 0), 3)
        
    # Plot Epipolar Lines on Image 2
    image2_with_matches = img_right.copy()
    for point in points2:
        x, y = map(int, [point[0], point[1]])
        cv2.circle(image2_with_matches, (x, y), 5, (0, 255, 0), 3)
    
    plt.figure('Matched SIFT Features')
    plt.subplot(121)
    plt.imshow(image1_with_matches)
    plt.subplot(122)
    plt.imshow(image2_with_matches)
    plt.savefig(output_path + 'Matched_SIFT.jpeg')
    plt.show()
    
    F, mask, E, R, t = compute_rotation_translation(points1, points2, calib_data_L['cameraMatrix'], calib_data_R['cameraMatrix'])
    
    # print("Rotation:\n", R)
    # print("Translation:\n", t)
    
    points1 = points1[mask.ravel()==1]
    points2 = points2[mask.ravel()==1]
    
    # Compute Epipolar Lines
    epilines1 = cv2.computeCorrespondEpilines(points2, 2, F)
    epilines1 = epilines1.reshape(-1, 3)

    epilines2 = cv2.computeCorrespondEpilines(points1, 1, F)
    epilines2 = epilines2.reshape(-1, 3)
    
    # Plot Epipolar Lines on Image 1
    image1_with_lines = img_left.copy()
    for line in epilines1:
        x0, y0, x1, y1 = map(int, [0, -line[2]/line[1], img_right.shape[1], -(line[2]+line[0]*img_right.shape[1])/line[1]])
        cv2.line(image1_with_lines, (x0, y0), (x1, y1), (0, 255, 0), 1)
        
    # Plot Epipolar Lines on Image 2
    image2_with_lines = img_right.copy()
    for line in epilines2:
        x0, y0, x1, y1 = map(int, [0, -line[2]/line[1], img_left.shape[1], -(line[2]+line[0]*img_left.shape[1])/line[1]])
        cv2.line(image2_with_lines, (x0, y0), (x1, y1), (0, 255, 0), 1)
    
    plt.figure('Epipolar Lines')
    plt.subplot(121)
    plt.imshow(image1_with_lines)
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio
    plt.subplot(122)
    plt.imshow(image2_with_lines)
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio
    plt.savefig(output_path + 'Epipolar_Lines.jpeg')
    plt.show()
    
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    
    print("Camera Matrix L:\n", calib_data_L['cameraMatrix'])
    print("Camera Matrix R:\n", calib_data_R['cameraMatrix'])
    print("Translation Vector:\n",t)
    print("Rotation Matrix:\n", R)
    
    image_size = img_right.shape[::-1]
    R1, R2, P1, P2, Q, roi_1, roi_2 = cv2.stereoRectify(calib_data_L['cameraMatrix'], calib_data_L['distCoeff'], calib_data_R['cameraMatrix'], calib_data_R['distCoeff'], image_size, R, t)
    mapx1, mapy1 = cv2.initUndistortRectifyMap(calib_data_L['cameraMatrix'], calib_data_L['distCoeff'], R1, P1, image_size, cv2.CV_32FC1)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(calib_data_R['cameraMatrix'], calib_data_R['distCoeff'], R2, P2, image_size, cv2.CV_32FC1)  
    print("Q1: \n", Q)
    
    rectified_img1 = cv2.remap(img_left, mapx1, mapy1, cv2.INTER_LINEAR)
    # x, y, w, h = roi_1
    # rectified_img1 = rectified_img1[y:y+h, x:x+w]
    
    rectified_img2 = cv2.remap(img_right, mapx2, mapy2, cv2.INTER_LINEAR)
    # x, y, w, h = roi_2
    # rectified_img2 = rectified_img2[y:y+h, x:x+w]
    
    plt.figure('Rectified Images')
    plt.subplot(121)
    plt.imshow(rectified_img1)
    plt.subplot(122)
    plt.imshow(rectified_img2)
    plt.savefig(output_path + 'Rectified_Images.jpeg')
    plt.show()
    
    window_size = 3
    # StereoSGBM parameters
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=16,
        numDisparities=5 * 16,
        blockSize=window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM
    )

    # Create a right matcher based on the left matcher
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Disparity computation
    displ = left_matcher.compute(img_left, img_right).astype(np.float32) / 16.0
    # plt.imshow(displ)
    # plt.colorbar()
    # plt.show()
    dispr = right_matcher.compute(img_right, img_left).astype(np.float32) / 16.0
    # plt.imshow(dispr)
    # plt.colorbar()
    # plt.show()

    # Disparity WLS filter
    lmbda = 8000
    sigma = 1.5
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disparity = wls_filter.filter(displ, img_left, None, dispr)
    
    # Display Disparity Map
    plt.figure('Disparity Map')
    plt.imshow(filtered_disparity)
    plt.colorbar()
    plt.savefig(output_path + 'Disparity_Map.jpeg')
    plt.show()
    
    print(filtered_disparity)
    
    # Identify invalid values (e.g., 0 or a specific value indicating invalid)
    invalid_mask = (filtered_disparity == 0)

    # Coordinates of valid and invalid points
    valid_coordinates = np.argwhere(~invalid_mask)
    invalid_coordinates = np.argwhere(invalid_mask)
    print(invalid_coordinates)

    # Extract values from valid points
    valid_values = filtered_disparity[~invalid_mask]

    # Perform interpolation to fill invalid values
    interpolated_values = griddata(valid_coordinates, valid_values, invalid_coordinates, method='linear')

    # Replace invalid values with interpolated values
    filtered_disparity[invalid_mask] = interpolated_values
    
    # depth = (fx * baseline) / (disparity + (cx2 - cx1))
    depth_map = ((calib_data_L['cameraMatrix'][0,0] * np.linalg.norm(t)) / (filtered_disparity))
    plt.figure('Depth Map')
    plt.imshow(depth_map)
    plt.colorbar()
    plt.savefig(output_path + '/Depth_Map.jpeg')
    plt.show()
     
    img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2RGB)
    
    intrinsic_params = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_params.intrinsic_matrix = calib_data_L['cameraMatrix']

    # create an rgbd image object:
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(img_left),
        o3d.geometry.Image(filtered_disparity),
        depth_scale=100.0, depth_trunc=200.0, convert_rgb_to_intensity=False)
    # use the rgbd image to create point cloud:
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_params)
    # visualize:
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(output_path + '/Point_Cloud.pcd', pcd)
 
if __name__ == '__main__':
    main()
