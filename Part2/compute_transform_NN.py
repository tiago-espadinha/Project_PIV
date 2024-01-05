import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from stereo_model import CREStereo

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
    
    # # Normalize the image coordinates
    # points1_normalized = cv2.undistortPoints(points1.reshape(-1, 2), K1, distCoeffs=None)
    # points2_normalized = cv2.undistortPoints(points2.reshape(-1, 2), K2, distCoeffs=None)
    
    # print(points1_normalized)
    # print(points2_normalized)

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
    
    # The code is importing the `imread_from_url` function from the `imread_from_url` module and using
    # it to load images from URLs. It then initializes a `CREStereo` object with a specified model
    # path and uses the `depth_estimator` object to estimate the depth map from the left and right
    # images. The shape of the left and right images is printed to the console.

    # Initialize model
    model_path = 'pre_trained_models/crestereo_combined_iter10_720x1280.onnx'
    depth_estimator = CREStereo(model_path)

    # Load images
    left_img = cv2.imread("my_datasets/trymefirst/quarto/left.jpeg")
    right_img = cv2.imread("my_datasets/trymefirst/quarto/right.jpeg")
    frameSize = (600, 800)
    left_img = cv2.resize(left_img, frameSize)
    right_img = cv2.resize(right_img, frameSize)
    
    # Detect keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(left_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(right_img, None)
    
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
    
    # Estimate depth and colorize it
    for i in range(1):
        disparity_map = depth_estimator(left_img, right_img)
    color_disparity = depth_estimator.draw_disparity()
    print(np.max(color_disparity), np.min(color_disparity))
    cv2.imwrite('disparity_map.png', color_disparity)
    
    print(disparity_map, disparity_map.shape)
    disparity_map = cv2.resize(disparity_map, frameSize)
    plt.imshow(disparity_map)
    plt.colorbar()
    plt.show()

    combined_img = np.hstack((left_img, color_disparity))

    cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)
    cv2.imshow("Estimated disparity", combined_img)
    cv2.waitKey(0)
    
    left_img = cv2.cvtColor(cv2.resize(left_img, frameSize), cv2.COLOR_BGR2RGB)
    
    calib_data_R = loadmat("chessBoard/calibration_parameters/calib_R.mat")
    calib_data_L = loadmat("chessBoard/calibration_parameters/calib_L.mat")
    
    F, mask, E, R, t = compute_rotation_translation(points1, points2, calib_data_L['cameraMatrix'], calib_data_R['cameraMatrix'])
    
    # Convert disparity to depth
    depth_map = depth_map = ((calib_data_L['cameraMatrix'][0,0] * np.linalg.norm(t)) / (disparity_map))
    depth_map = cv2.normalize(depth_map, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    
    plt.imshow(depth_map)
    plt.colorbar()
    plt.show()
    
    intrinsic_params = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_params.intrinsic_matrix = calib_data_L['cameraMatrix']

    # create an rgbd image object:
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(left_img),
        o3d.geometry.Image(depth_map),
        depth_scale=1.0, depth_trunc=1.0, convert_rgb_to_intensity=False)
    # use the rgbd image to create point cloud:
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_params)
    # visualize:
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
    
if __name__ == '__main__':
    main()

