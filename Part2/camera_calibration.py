import numpy as np
import cv2 as cv
import glob
from scipy.io import savemat
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

class Camera:
    
    def __init__(self, frameSize, criteria, images, calibration_data_output):
        self.frameSize = frameSize
        self.criteria = criteria
        self.images = images
        self.calibration_data = {}
        self.calibration_data_output = calibration_data_output
        return
    
    def calibrate(self, objp, chessboardSize):
        
        objpoints = []
        imgpoints = []
        
        for image in self.images:
            img = cv.imread(image)
            img = cv.resize(img, self.frameSize)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = cv.resize(gray, self.frameSize)
            cv.imshow('gray', gray)
            cv.waitKey(1000)
            
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(1000)


        cv.destroyAllWindows()

        ############## CALIBRATION #######################################################

        ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, self.frameSize, None, None)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        self.calibration_data = {
            'cameraMatrix': cameraMatrix,
            'distCoeff': dist,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'objpoints': objpoints,
            'imgpoints': imgpoints
        }

        savemat(self.calibration_data_output, self.calibration_data)
        return objpoints
        
    def undistort(self, objpoints):
        ############## UNDISTORTION #####################################################

        img = cv.imread(self.images[0])
        img = cv.resize(img, self.frameSize)
        h,  w = img.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(self.calibration_data['cameraMatrix'], self.calibration_data['distCoeff'], (w,h), 1, (w,h))



        # Undistort
        dst = cv.undistort(img, self.calibration_data['cameraMatrix'], self.calibration_data['distCoeff'], None, newCameraMatrix)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite('caliResult1.png', dst)



        # Undistort with Remapping
        mapx, mapy = cv.initUndistortRectifyMap(self.calibration_data['cameraMatrix'], self.calibration_data['distCoeff'], None, newCameraMatrix, (w,h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite('caliResult2.png', dst)

        # Reprojection Error
        mean_error = 0

        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], self.calibration_data['rvecs'][i], self.calibration_data['tvecs'][i], self.calibration_data['cameraMatrix'], self.calibration_data['distCoeff'])
            error = cv.norm(self.calibration_data['imgpoints'][i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error

        print( "total error: {}".format(mean_error/len(objpoints)))
        return
                    
def main():

    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
    
    # Load config file paths
    config = load_config('conf_file.cfg', 'DEFAULT')
    calib_path = config['calib_path']
    calib_path_in = calib_path + '/input/'
    calib_path_out = calib_path + '/output/'
    path_exists(calib_path_out)

    chessboardSize = (9, 6)
    frameSize = (600, 800)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 25
    objp = objp * size_of_chessboard_squares_mm


    # Arrays to store object points and image points from all the images.
    camera_R = Camera(frameSize, criteria, glob.glob(calib_path_in + 'images_R/*.jpeg'), calib_path_out + 'calib_R.mat')
    objpoints = camera_R.calibrate(objp, chessboardSize)
    camera_L = Camera(frameSize, criteria, glob.glob(calib_path_in + 'images_L/*.jpeg'),  calib_path_out + 'calib_L.mat')
    objpoints = camera_L.calibrate(objp, chessboardSize)
    
if __name__ == '__main__':
    main()