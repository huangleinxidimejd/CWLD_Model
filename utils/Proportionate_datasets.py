import os
import shutil
import random

"""Split the dataset proportionally"""

# Define the path to the source folder and three subfolders
src_dir = 'E:/CWLD_model/data/all_data/images'

new_train = 'E:/CWLD_model/data/train/images/'
new_val = 'E:/CWLD_model/data/val/images/'
new_test = 'E:/CWLD_model/data/test/images/'

# Create three subfolders
os.makedirs(new_train, exist_ok=True)
os.makedirs(new_val, exist_ok=True)
# os.makedirs(new_test, exist_ok=True)

# Get the paths of all files in the source folder
files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

# Division of documents in accordance with the ratio 8:1:1
num_files = len(files)

num_files1 = int(num_files * 0.8)
num_files2 = int(num_files * 0.1)
num_files3 = num_files - num_files1 - num_files2

# Randomly disrupting the file list
random.shuffle(files)

# Copy files into three subfolders
for i in range(num_files1):
    shutil.copy(files[i], new_train)

for i in range(num_files1, num_files1 + num_files2):
    shutil.copy(files[i], new_val)

for i in range(num_files1 + num_files2, num_files):
    shutil.copy(files[i], new_test)
