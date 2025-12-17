import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(2,2), feature_vector=True)

def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, 24, 3, method="uniform")
    hist, _ = np.histogram(lbp, bins=26, range=(0,26), density=True)
    return hist

def build_gabor_bank():
    kernels = []
    for theta in np.arange(0, np.pi, np.pi/8):
        k = cv2.getGaborKernel((31,31), 4.0, theta, 10.0, 0.5, 0)
        kernels.append(k)
    return kernels

def extract_gabor(image, kernels):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    feats = []
    for k in kernels:
        f = cv2.filter2D(gray, cv2.CV_32F, k)
        feats.extend([f.mean(), f.std(), np.abs(f).max()])
    return np.array(feats)

def extract_combined(image):
    return np.hstack([extract_hog(image), extract_lbp(image)])
