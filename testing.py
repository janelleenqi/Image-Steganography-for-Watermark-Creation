import cv2 # OpenCV
import numpy as np # Arrays (1D, 2D, and matrices)
import matplotlib.pyplot as plt # Plots
import math

def read_image(imgPath):
    image = cv2.imread(imgPath) #3 channel rgb image
    return image

def save_img(image, imageDesc):
    cv2.imwrite(str(f"{imageDesc}.png"), image)

testImg = read_image('WatermarkEncodedImg.png')
height, width = testImg.shape[:2]

# resized image
scale = 0.999
resizedEmbeddedImg = cv2.resize(testImg, (int(width * scale), int(height * scale)))
save_img(resizedEmbeddedImg, 'camera_watermarkedPeguin_resized')

# crop image
croppedEmbeddedImg = testImg[1:height-2, 1:width-2] # crop outermost pixel row and column
save_img(croppedEmbeddedImg, 'camera_watermarkedPeguin_cropped')

# rotate image
center = (width // 2, height // 2)
M = cv2.getRotationMatrix2D(center, -2, 1.0)  # Create rotation matrix: negative angle => counterclockwise (tilt left), -2 degrees, scale = 1.0
rotatedEmbeddedImg = cv2.warpAffine(testImg, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
save_img(rotatedEmbeddedImg, 'camera_watermarkedPeguin_rotated')
