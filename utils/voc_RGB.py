import os

import numpy as np
from PIL import Image
from skimage import io

import os
from PIL import Image

"""Calculate the number of RGB pixels in the image"""

# Open image
folder_path = 'E:/'

# Get the paths of all image files in a folder
image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.png'))]
all_blue_count = 0
all_red_count = 0
all_black_count = 0
all_white_count = 0

# Batch Read Images
images = []
for image_path in image_paths:
    image = Image.open(image_path)
    images.append(image)

# Print the number of images read
print(f"A total of {len(images)} images have been read.")


for image in images:
    # Pixel value to be counted
    blue = (0, 0, 255)
    red = (255, 0, 0)
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Convert the image to RGB mode (if not)
    image = image.convert('RGB')

    # Counting the number of pixel values
    blue_count = 0
    red_count = 0
    black_count = 0
    white_count = 0
    width, height = image.size
    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            if pixel == blue:
                blue_count += 1
            if pixel == red:
                red_count += 1
            if pixel == black:
                black_count += 1
            if pixel == white:
                white_count += 1

    print(f"Number of blue: {blue_count}")
    print(f"Number of red: {red_count}")
    print(f"Number of black: {black_count}")
    print(f"Number of white: {white_count}")
    all_blue_count = all_blue_count + blue_count
    all_red_count = all_red_count + red_count
    all_black_count = all_black_count + black_count
    all_white_count = all_white_count + white_count
print("---------------------------------------")
print(f"Total blue: {all_blue_count}")
print(f"Total red: {all_red_count}")
print(f"Total Black: {all_black_count}")
print(f"Total White: {all_white_count}")
print(all_blue_count+all_red_count+all_black_count+all_white_count)

file_path = 'E:/XX.txt'
f = open(file_path, 'w')
f.write('all_blue_count %s\n' % all_blue_count)
f.write('all_red_count %s\n' % all_red_count)
f.write('all_black_count %s\n' % all_black_count)
f.write('all_white_count %s\n' % all_white_count)
