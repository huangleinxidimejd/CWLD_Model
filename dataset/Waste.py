import os
import cv2
import torch
import numpy as np
from skimage import io
from torch.utils import data
import tifffile as tiff
# from osgeo import gdal_array
import matplotlib.pyplot as plt
import utils.transform as transform
from skimage.transform import rescale
from torchvision.transforms import functional as F

num_classes = 4
COLORMAP = [[125, 125, 125], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255]]
CLASSES = ['Invalid', 'Background', 'Empty', 'Waste', 'Facilities']

"""
MEAN: the mean of the three channels of the training dataset RGB.
STD: standard deviation of the three channels of the training dataset RGB.
"""
MEAN = np.array([121.33, 128.66, 120.39])
STD = np.array([64.21, 61.32, 60.84])

# Data storage path
root = 'F:/HL/WasteSeg/data'


def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


# Normalize a single image
def normalize_image(im):
    return (im - MEAN) / STD


# Normalize multiple images
def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs


"""
Create a color-indexed lookup table 
0 [125, 125, 125]
1 [0, 0, 0]
2 [255, 255, 255]
3 [255, 0, 0]
4 [0, 0, 255]
"""
colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


# Convert index label images to color label images
def Index2Color(pred):
    colormap = np.asarray(COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


# Convert multiple color label images to indexed images
def Colorls2Index(ColorLabels):
    for i, data in enumerate(ColorLabels):
        ColorLabels[i] = Color2Index(data)
    return ColorLabels


# Convert single color label images to indexed images (classification)
def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    # IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap <= num_classes)
    return IndexMap.astype(np.uint8)


# Scale multiple images to a specified ratio and interpolation method.
def rescale_images(imgs, scale, order):
    for i, im in enumerate(imgs):
        imgs[i] = rescale_image(im, scale, order)
    return imgs


# Scales a single image to a specified ratio and interpolation method.
def rescale_image(img, scale=1 / 8, order=0):
    flag = cv2.INTER_NEAREST
    if order == 1:
        flag = cv2.INTER_LINEAR
    elif order == 2:
        flag = cv2.INTER_AREA
    elif order > 2:
        flag = cv2.INTER_CUBIC
    im_rescaled = cv2.resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)),
                             interpolation=flag)
    return im_rescaled


# Get a list of file names in the dataset
def get_file_name(mode):
    data_dir = root
    # assert mode in ['train', 'test']
    mask_dir = os.path.join(data_dir, mode, 'images')

    data_list = os.listdir(mask_dir)
    for vi, it in enumerate(data_list):
        data_list[vi] = it[:-4]
    return data_list


# Read remote sensing images and tagged images
def read_RSimages(mode, rescale=False, rotate_aug=False):
    data_dir = root
    # assert mode in ['train', 'test']
    img_dir = os.path.join(data_dir, mode, 'images')
    mask_dir = os.path.join(data_dir, mode, 'label')

    data_list = os.listdir(img_dir)

    data, labels = [], []
    count = 0
    for it in data_list:
        it_name = it[:-4]
        it_ext = it[-4:]
        if (it_name[0]=='.'):
            continue
        if (it_ext == '.tif'):
            img_path = os.path.join(img_dir, it)

            # Select the corresponding function according to the label image format
            mask_path = os.path.join(mask_dir, it_name + '.png')
            # mask_path = os.path.join(mask_dir, it_name + '.tif')

            img = io.imread(img_path)
            # label = gdal_array.LoadFile(mask_path)

            label = io.imread(mask_path)
            # label = tiff.imread(mask_path)

            data.append(img)
            labels.append(label)

            count += 1
            if not count % 500: print('%d/%d images loaded.' % (count, len(data_list)))
            # if count: break
    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')

    return data, labels


# Read remote sensing images but no labeled images
def read_RSimages_nolabel(mode, rescale=False, rotate_aug=False):
    data_dir = root
    # assert mode in ['train', 'test']

    img_dir = os.path.join(data_dir, mode, 'images')

    data_list = os.listdir(img_dir)

    data = []
    count = 0
    for it in data_list:
        it_name = it[:-4]
        it_ext = it[-4:]
        if (it_name[0]=='.'):
            continue
        if (it_ext == '.tif'):
            img_path = os.path.join(img_dir, it)
            img = io.imread(img_path)
            data.append(img)
            count += 1
            if not count % 500: print('%d/%d images loaded.' % (count, len(data_list)))
            # if count: break
    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')
    return data


class RS(data.Dataset):
    """
    __init__ method: this method initializes the dataset class and receives two arguments, mode indicating the dataset
    mode and random_flip indicating whether or not random flip enhancement is performed.
    The method loads the dataset by calling the read_RSimages function.
    """

    def __init__(self, mode, random_flip=False):
        self.mode = mode
        self.random_flip = random_flip
        data, labels = read_RSimages(mode, rescale=False)

        self.data = data
        self.labels = Colorls2Index(labels)

        self.len = len(self.data)

    """
    __getitem__ method: this method defines the way to fetch the data in the dataset, 
    and receives a parameter idx, which indicates the data index in the dataset.The method first fetches the image 
    and label data corresponding to that index, then performs random flip enhancement 
    according to the random_flip parameter, then normalizes the image data and converts it to a tensor, and
    Finally, it returns the data and labels.
    """

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.random_flip:
            data, label = transform.rand_flip(data, label)
            # data = transform.rand_flip(data)
        data = normalize_image(data)
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        return data, label
        # return data

    """
    __len__ method: This method returns the length of the dataset, i.e. the number of pieces of data in the dataset.
    """
    def __len__(self):
        return self.len
