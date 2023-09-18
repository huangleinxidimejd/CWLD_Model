import cv2
import os.path
import tifffile as tiff
from skimage import io

# Image folder path
images_dir = r'F:/CWLD_model/data/val/images/'
label_dir = r'F:/CWLD_model/data/val/label/'
new_images = r'F:/CWLD_model/data/new_val/images/'
new_label = r'F:/CWLD_model/data/new_val/label/'



for img_name in os.listdir(images_dir):
    name = img_name[0:-4]
    # Image and label address
    img_path = images_dir + name + '.tif'
    label_path = label_dir + name + '.png'

    img = tiff.imread(img_path)
    # label = tiff.imread(label_path)
    label = io.imread(label_path)
    """Crop an area and scale it"""

    # img.shape(512, 512, 3)
    width, height, channel = img.shape

    crop_width = 480
    crop_height = 480

    # Define crop position coordinates
    left = int((width - crop_width)/2)
    top = int((height - crop_height)/2)
    right = left + crop_width
    bottom = top + crop_height

    cropped_image = img[top:bottom, left:right]
    cropped_label = label[top:bottom, left:right]


    # Saving scaled images
    tiff.imwrite(new_images + img_name[0:-4]+'.tif', cropped_image)
    io.imsave(new_label + img_name[0:-4]+'.png', cropped_label)