import cv2
import numpy as np

# shortcut for rgb to bgr transformation
def rgb_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# shortcut for bgr to rgb transformation
def bgr_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# shortcut for bgr to gray transformation
def bgr_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# shortcut for bgr to hls transformation
def bgr_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# shortcut for bgr to hsv transformation
def bgr_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lane parameter transformation: 
# polynomial coefficients => parameters
# w: width of the image in meters
# y_m: vertical position of the car in meters
def toParam(left_coeff, right_coeff, w=1280*3.7/700, y_m=720*3/50):
    # computing the curvature radii
    left_curverad_pm = ((1 + (2*left_coeff[0]*y_m + left_coeff[1])**2)**1.5) / (2*left_coeff[0])
    right_curverad_pm = ((1 + (2*right_coeff[0]*y_m + right_coeff[1])**2)**1.5) / (2*right_coeff[0])
    # computing the car's position
    pos_left = left_coeff[0]*y_m**2 + left_coeff[1]*y_m + left_coeff[2]
    pos_right = right_coeff[0]*y_m**2 + right_coeff[1]*y_m + right_coeff[2]
    pos_mid = 0.5 * (pos_right + pos_left)
    # difference to center of the image
    pos_car = w/2 - pos_mid
    # lane distance
    dist_lanes = pos_right - pos_left
    # angles
    alpha_left  = np.arcsin(2*left_coeff[0]*y_m + left_coeff[1])
    alpha_right = np.arcsin(2*right_coeff[0]*y_m + right_coeff[1])
    return pos_car, dist_lanes, alpha_left, alpha_right, left_curverad_pm, right_curverad_pm

# lane parameter transformation: 
# polynomial coefficients <= parameters
# w: width of the image in meters
# y_m: vertical position of the car in meters
def toCoeff(pos_car, dist_lanes, alpha_left, alpha_right, left_curverad_pm, right_curverad_pm, w=1280*3.7/700, y_m=720*3/50):
    sal=np.sin(alpha_left)
    sar=np.sin(alpha_right)
    A_left  = (1+sal**2)**1.5/(2*left_curverad_pm)
    A_right = (1+sar**2)**1.5/(2*right_curverad_pm)
    B_left  = sal - 2*A_left*y_m
    B_right = sar - 2*A_right*y_m
    C_left  = 0.5*(w - dist_lanes) - pos_car - A_left *y_m**2 - B_left*y_m
    C_right = 0.5*(w + dist_lanes) - pos_car - A_right*y_m**2 - B_right*y_m
    left_coeff  = np.array([A_left,  B_left,  C_left])
    right_coeff = np.array([A_right, B_right, C_right])
    return left_coeff, right_coeff




