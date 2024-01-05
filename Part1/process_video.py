import numpy as np
import scipy
from configparser import ConfigParser
import cv2
import time
import sys
import os


# Create a new folder if it doesn't exist
def path_exists(file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def main():

    # Opens the config file
    if len(sys.argv) != 2:
        config_file = 'conf_file.cfg'
    else:
        config_file = sys.argv[1]

    config = ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    undersampling_factor = int(config['undersampling_factor'])
    output_path = config['frames_directory']
    path_exists(output_path)

    # Load video
    video = cv2.VideoCapture(config['videos'])
    sift = cv2.SIFT_create(nfeatures = 0)
    video_data = []
    
    # Initialize a frame counter
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Identify frame keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(frame, None)
        if frame_count % undersampling_factor == 0:
            # Save the frame as an image
            frame_filename = f"{output_path}/frame_{int(frame_count/undersampling_factor):04d}.jpg"
            cv2.imwrite(frame_filename, frame)
            
            if frame_count == 0:
                frame_filename = f"{output_path}/map.jpg"
                cv2.imwrite(frame_filename, frame)
            
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
