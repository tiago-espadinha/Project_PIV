import numpy as np
import scipy
from configparser import ConfigParser
import cv2
import time
import sys

def main():
    # # Use config file as input argument
    # if len(sys.argv) != 2:
    #     print("Usage: python process_video.py <config_file>")
    #     sys.exit(1)

    # # Opens the config file
    # config_file = sys.argv[1]

    config_file = 'part1.cfg'

    config = ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']

    # Load video
    video = cv2.VideoCapture(config['videos'])
    sift = cv2.SIFT_create(nfeatures = 0)
    video_data = []
    
    # Set the undersampling factor
    undersampling_factor = 30  # Change this to the desired factor
    
    # Initialize a frame counter
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Identify frame keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(frame, None)
        if frame_count % undersampling_factor == 0:
            # Display the resulting frame
            img = cv2.drawKeypoints(frame, keypoints, frame)
            cv2.imshow('frame', img)

            # Append frame data to matlab array
            keypoints = np.float32([keypoints[m].pt for m in range(len(keypoints))]).reshape(-1,2)
            descriptors = np.float32(descriptors)
            frame_data = {'keypoints': keypoints, 'descriptors': descriptors}
            video_data.append(frame_data)
            
        # Increment the frame counter
        frame_count += 1

        # time.sleep(1/30)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    # Prepare data for matlab
    compiled_data = np.array([(frame_data['keypoints'], frame_data['descriptors']) for frame_data in video_data], dtype=object)
    mdic = {"features": compiled_data}
    scipy.io.savemat(config['keypoints_out'], mdic)

    # Check output format
    f=scipy.io.loadmat(config['keypoints_out'])
    feat = f['features']
    print("Feature")
    print(" -> type: ", type(feat))
    print(" -> shape: ", feat.shape)

if __name__ == '__main__':
    main()
