import os
import numpy as np
import tifffile as tiff

"""
Calculate the mean and std of the data set
"""
# Define the path to the training dataset

train_data_path = 'F:/HL/CWLD_model/data/train/images'
files = os.listdir(train_data_path)

# Load all training images and store them in an array or list
train_data = []
for img_name in files:
    # print(img_name)
    # img = cv2.imread(os.path.join(train_data_path, img_name))
    img = tiff.imread(os.path.join(train_data_path, img_name))
    train_data.append(img)

print("%d data load completed！" % len(files))

train_data = np.array(train_data)

# Calculate the mean value for each channel
mean_channel_1 = np.mean([img[:, :, 0] for img in train_data])
mean_channel_2 = np.mean([img[:, :, 1] for img in train_data])
mean_channel_3 = np.mean([img[:, :, 2] for img in train_data])
print("Mean value calculation complete！")

# Calculate the standard deviation for each channel
std_channel_1 = np.std([img[:, :, 0] for img in train_data])
std_channel_2 = np.std([img[:, :, 1] for img in train_data])
std_channel_3 = np.std([img[:, :, 2] for img in train_data])
print("Standard deviation calculation completed！")

# Combine the means of the three channels into an RGB mean vector
mean = [mean_channel_1, mean_channel_2, mean_channel_3]
std = [std_channel_1, std_channel_2, std_channel_3]

# Print results
print("RGB MEA：", mean)
print("RGB STD：", std)

with open('mean_std.txt', 'a') as file0:
    print('mean:%s' % mean, 'std:%s' % std, file=file0)
