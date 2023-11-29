import numpy as np
import scipy

def main():

    # what should be outputed
    f=scipy.io.loadmat('Project_PIV/specs/surf_features.mat')
    feat = f['features']
    print("Feature")
    print(" -> type: ", type(feat))
    print(" -> shape: ", feat.shape)
    print("Keypoint")
    print(" -> shape: ", feat[0,2].shape)

if __name__ == '__main__':
    main()
    