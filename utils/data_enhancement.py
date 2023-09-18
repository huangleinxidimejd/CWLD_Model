import random
import time
import cv2
import numpy as np
import os.path
import tifffile as tiff
from skimage import io

"""
Data enhancement methods
"""

# peppercorn noise
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# Gaussian noise
def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# 50% darker
def darker(image, percetage=0.6):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# 50% brighter
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


# revolve
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the images is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# flips
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image


start = time.time()
# Image folder path
images_dir = r'E:/CWLD_model/data/all_data/images/'
label_dir = r'E:/CWLD_model/data/all_data/label/'

for img_name in os.listdir(images_dir):
    name = img_name[0:-4]
    # Image and label address
    img_path = images_dir + name + '.tif'
    label_path = label_dir + name + '.png'

    img = tiff.imread(img_path)
    # label = tiff.imread(label_path)
    label = io.imread(label_path)

    # # revolve
    # img_rotated_90 = rotate(img, 90)
    # label_rotated_90 = rotate(label, 90)

    # mirroring
    flipped_img = flip(img)
    flipped_label = flip(label)
    tiff.imwrite(images_dir + img_name[0:-4] + '_fli.tif', flipped_img)
    io.imsave(label_dir + img_name[0:-4] + '_fli.png', flipped_label)

    # Adding noise (usually adding Gaussian noise)
    # img_salt = SaltAndPepper(img, 0.3)
    # tiff.imwrite(images_dir + img_name[0:7] + '_salt.tif', img_salt)
    img_gauss = addGaussianNoise(img, 0.3)
    label_gauss = label
    tiff.imwrite(images_dir + img_name[0:-4] + '_noise.tif', img_gauss)
    io.imsave(label_dir + img_name[0:-4] + '_noise.png', label_gauss)

    # blur = cv2.GaussianBlur(img, (7, 7    ), 1.5)
    # cv2.imwrite(images_dir + img_name[0:-4] + '_blur.jpg', blur)

    # 50% darker
    img_darker = darker(img)
    label_darker = label
    tiff.imwrite(images_dir + img_name[0:-4] + '_darker.tif', img_darker)
    io.imsave(label_dir + img_name[0:-4] + '_darker.png', label_darker)

    # 50% brighter
    img_brighter = brighter(img)
    label_brighter = label
    tiff.imwrite(images_dir + img_name[0:-4] + '_brighter.tif', img_brighter)
    io.imsave(label_dir + img_name[0:-4] + '_brighter.png', label_brighter)

    # resizing
    """Crop a random area and zoom in and out"""
    crop_width = 300
    crop_height = 300

    # img.shape(512, 512, 3)
    width, height, channel = img.shape

    # Define crop position coordinates
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height

    # Cropping images and labeling images
    cropped_image = img[top:bottom, left:right]
    cropped_label = label[top:bottom, left:right]

    # cv2.resize() function can be used to resize JPEG, PNG, BMP, TIFF, PBM, PGM, PPM and more!
    zoom_image = cv2.resize(cropped_image, (width, height))
    zoom_label = cv2.resize(cropped_label, (width, height))

    # Save scaled images
    tiff.imwrite(images_dir + img_name[0:-4] + '_zoom.tif', zoom_image)
    io.imsave(label_dir + img_name[0:-4] + '_zoom.png', zoom_label)

end = time.time()
print(end - start)
