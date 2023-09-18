import os
import tifffile as tiff
from skimage import io

"""Copying files with the same name from a folder to a new folder"""

# Define the path to the source folder and three subfolders
src_dir = 'E:/CWLD_model/data/all_data/label/'

new_train_images = 'E:/CWLD_model/data/train/images/'
new_train_label = 'E:/CWLD_model/data/train/label/'

new_val_images = 'E:/CWLD_model/data/val/images/'
new_val_label = 'E:/CWLD_model/data/val/label/'


for img_name in os.listdir(new_train_images):
    # Label image address
    name = img_name[0:-4]

    label_path = src_dir + name + '.png'
    label = io.imread(label_path)
    io.imsave(new_train_label + name + '.png', label)

for img_name in os.listdir(new_val_images):
    # Label image address
    name = img_name[0:-4]

    label_path = src_dir + name + '.png'
    label = io.imread(label_path)
    io.imsave(new_val_label + name + '.png', label)

# for img_name in os.listdir(new_test_images):
#     # Label image address
#     label_path = src_dir + img_name
#     label = tiff.imread(label_path)
#     tiff.imwrite(new_test_label + img_name[0:-4] + '.tif', label)
