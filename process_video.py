import numpy as np
import scipy
import sys
from configparser import ConfigParser
import cv2
import time

def main():
    # Use config file as input argument
    # if len(sys.argv) != 2:
    #     print("Usage: python process_video.py <config_file>")
    #     sys.exit(1)

    # # Opens the config file
    # config_file = sys.argv[1]


    config_file = 'conf_file.cfg'
    config = ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']

    print("Config file: ", config['videos'])
    video = cv2.VideoCapture(config['videos'])

    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        time.sleep(1/30)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # What should be outputed
    f=scipy.io.loadmat('Project_PIV/specs/surf_features.mat')
    feat = f['features']
    print("Feature")
    print(" -> type: ", type(feat))
    print(" -> shape: ", feat.shape)
    print("Keypoint")
    print(" -> shape: ", feat[0,2].shape)

if __name__ == '__main__':
    main()