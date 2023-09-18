import time
import cv2
import os.path
import tifffile as tiff
from skimage import io

"""
Changing the image size
"""
start = time.time()

# Image folder path
images_dir = r'F:/HL/WasteSeg/data/train/images/'
label_dir = r'F:/HL/WasteSeg/data/train/label/'
new_images_dir = r'F:/HL/WasteSeg/data/train/new_images/'
new_label_dir = r'F:/HL/WasteSeg/data/train/new_label/'
for img_name in os.listdir(images_dir):
    name = img_name[0:-4]
    # Image and label address
    img_path = images_dir + name + '.tif'
    label_path = label_dir + name + '.png'

    img = tiff.imread(img_path)
    label = io.imread(label_path)

    width = 512
    height = 512

    # cv2.resize() function can be used to adjust the JPEG, PNG, BMP, TIFF, PBM, PGM, PPM and other
    zoom_image = cv2.resize(img, (width, height))
    zoom_label = cv2.resize(label, (width, height))

    # Saving the adjusted image
    tiff.imwrite(new_images_dir + img_name[0:-4] + '.tif', zoom_image)
    io.imsave(new_label_dir + img_name[0:-4] + '.png', zoom_label)

end = time.time()
print(end - start)
