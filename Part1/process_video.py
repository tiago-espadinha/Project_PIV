import numpy as np
import scipy.io as sio
import cv2
import sys
import os

# Load config file
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


def main():

    # Opens the config file
    if len(sys.argv) != 2:
        config_file = 'conf_file.cfg'
    else:
        config_file = sys.argv[1]

    config = parse_config_file(config_file)

    undersampling_factor = 1
    # image_map = config['image_map'][0]

    # Load video
    video = cv2.VideoCapture(config['videos'][0])
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
            # if frame_count == 0:
            #     cv2.imwrite(image_map, frame)
            
            # Display the resulting frame
            img = cv2.drawKeypoints(frame, keypoints, frame)
            #cv2.imshow('frame', img)

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
    sio.savemat(config['keypoints_out'][0], mdic)

    # Check output format
    # f=sio.loadmat(config['keypoints_out'][0])
    # feat = f['features']
    # print("Feature")
    # print(" -> type: ", type(feat))
    # print(" -> shape: ", feat.shape)

if __name__ == '__main__':
    main()

