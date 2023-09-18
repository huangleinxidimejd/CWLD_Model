import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

VOCdevkit_path = 'F:/CWLD_model/data/pred/'

if __name__ == "__main__":
    random.seed(0)

    temp_seg = os.listdir(VOCdevkit_path)
    total_seg = []
    for seg in temp_seg:
        total_seg.append(seg)

    num = len(total_seg)
    list = range(num)

    classes_nums = np.zeros([256], np.int)
    for i in tqdm(list):
        name = total_seg[i]
        png_file_name = os.path.join(VOCdevkit_path, name)
        if not os.path.exists(png_file_name):
            raise ValueError("Tagged image %s not detected, please check if the file exists in the specific path and "
                             "if the suffix is png" % (png_file_name))

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("The label image %s has a shape of %s and is not a grayscale or eight-bit color image, so please "
                  "double-check the dataset format." % (name, str(np.shape(png))))
            print("The label image needs to be either grayscale or eight-bit color, and the value of each pixel point "
                  "of the label is the category to which the pixel point belongs." % (name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)
