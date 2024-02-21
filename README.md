# CWLD_Model
* DeepLabV3+ network is a multi-scale, multipath parallel convolutional neural network proposed by Chen,L.C et al. in 2018.The main advantage of DeepLabV3+ network is its excellent image segmentation performance and robustness. Its core innovation is the introduction of an encoding-decoding structure that combines low-level semantic information with high-level semantic information, thus improving the segmentation accuracy of the network. We have improved the DeepLabV3+ network by dividing it into four main components: the backbone network, the Atrous Spatial Pyramid Pooling (ASPP) module, the encoding structure, and the decoding structure. The backbone network uses ResNet-101 with a residual structure for feature extraction from the original image, and the ASPP module is a null convolution with different expansion rates for fusion of features at different scales, which increases the sensory field without losing image information. The coding structure is used for feature extraction while reducing the size of the feature map to reduce the computational complexity. The decoding structure is used for up-sampling to recover the spatial detail information of the image, and for fusion of deep and shallow features to obtain finer recognition results.
![网络模型](https://github.com/huangleinxidimejd/CWLD_Model/assets/42790126/228cbb85-ab7b-4330-a600-7dec89192857)

## Data
The dataset used to train the model is the CWLD dataset published on Zenodo.
https://zenodo.org/record/8333888

## Training details
The model was trained using two GPUs Nvidia GeForce RTX 2080Ti and the following parameters:  
   * 'train_batch_size': 4,  
   * 'val_batch_size': 4,  
   * 'train_crop_size': 512,  
   * 'val_crop_size': 512,  
   * 'lr': 0.001,  # The learning rate used during training. It determines how fast the model learns from the data  
   * 'epochs': 200,  
   * 'gpu': True,  
   * 'weight_decay': 5e-4,  
   * 'momentum': 0.9,  
   * 'print_freq': 100,  
   * 'predict_step': 5,  
## Usage
* After downloading the dataset from Zenodo, Place the train and val files from the Deep Learning Datasets file into the data folder of the CWLD semantic segmentation model.  
* Open: CWLD_ Open the root in: CWLD_model/dataset/ and start training with the WasteSeg_Train.py file. The modelss module provides five kinds of convolutional networks: Improved_DeeplabV3_plus, PSPNet, ResNet, SegNet and UNet, which can be selected and modified accordingly.  
* The utils package provides a large number of data processing tools for use.  
* The trained model can be predicted by EvalSeg.py file.

## Detailed description of utils package
* Acc_F1_chart.py

      The main purpose of this code is to plot the F1 scores (accuracy) curves of four neural network models (Deeplabv3, PSPNet, SegNet, UNet) at different number of iterations (Epoch).
      First, it creates a list x_all containing 200 elements,and then uses the zip() function to pair x_all with the F1 scores of the four models, stored in line1, line2, line3, and line4.
      Next, use the matplotlib library to plot these four curves, using different colors, markers, line types, and line widths for each curve. Finally, add a legend for each curve and save the plotted image as a file named "F1.png".
* calculat_mean_std.py

      The main purpose of this code is to compute the mean and standard deviation of the RGB channels of a set of training images.
      First, it imports the required libraries, including os, numpy, and tifffile. then, it defines the path to the training dataset and gets the names of all the files under that path.
      Next, by traversing these filenames, each image file is read using the tifffile library and they are stored in a list called train_data.
      After loading all the training images, train_data is converted to a numpy array. Then, the mean and standard deviation of each channel (red, green, and blue) are calculated separately.
      Finally, the mean values of the three channels are combined into an RGB mean vector and printed out along with the standard deviation. Also, the results are written to a file named 'mean_std.txt'.

* change_size.py

      The main function of this code is to resize images.
      It first imports the required library and then defines the paths to the image folder and the label folder.
      Next, it iterates through all the files in the image folder and reads each image and the corresponding label file.
      It then sets the width and height of the images and labels to 512 pixels and resizes them using the cv2.resize() function.
      Finally, it saves the resized images and labels to a new folder and prints out the time spent on the whole process.

* Copyfile_with_SameName.py

      The main function of this code is to copy files with the same name from one folder to another new folder. 
      Specifically, it first defines the paths of the source folder and the three subfolders, then iterates through the image files in the training set and validation set,
      reads the labels (i.e., PNG files with the same name as the image files) of each image file, and copies these labeled files to the corresponding labeled folders of the training set and validation set.

* Cropping_images.py

      The main function of this code is to read the images and corresponding labels in a folder, then crop and scale the images, and finally save the processed images and labels to a new folder.
      The specific steps are as follows:
      * Import the required libraries: cv2, os.path, tifffile (alias tiff) and the io module of skimage.
      * Define the image folder path, the label folder path, and the new image folder path and new label folder path.
      * Use the os.listdir() function to iterate through all the filenames in the image folder.
      * For each filename, remove the suffix (assuming '.tif'), to get the filename (excluding the suffix).
      * Construct the image file path and label file path based on the filename.
      * Use imread() function of tifffile to read image file, and imread() function of io module of skimage to read label file.
      * Get the width, height and number of channels of the image. Define the width and height of the cropping area (480 pixels).
      * Calculate the coordinates of the upper left corner (left, top) and the lower right corner (right, bottom) of the cropping area.
      * Crop the image and label to get the cropped image (cropped_image) and label (cropped_label).
      * Use imwrite() function of tifffile to save the cropped image to a new image folder, and use imsave() function of io module of skimage to save the cropped label to a new label folder.


* data_enhancement.py

      This code mainly implements the following functions:
      * Reads all the image and label files in the images and labels folder.
      * Performs a flip operation on an image.
      * Performs pretzel noise addition to the image.
      * Performs Gaussian noise addition to the image.
      * Performs a 50% darken operation on an image.
      * Performs a 50% lighten operation on an image.
      * Rotate an image.
      * Performs a crop operation on the image by randomly selecting an area to crop and then zooming in and out to the original size.
      * Save the processed image and label to the appropriate folder.

* eval.py

      This code defines two functions, eval_net and eval_net_BCE, which are used to evaluate the performance of a neural network model. Both functions take three arguments: net (the neural network model), dataset (the dataset), and gpu (whether or not to use a GPU, which defaults to True).
      The eval_net function calculates the average loss value of the model on the given dataset and returns that value. It first sets the model to evaluation mode (net.eval()), and then iterates through each sample in the dataset. For each sample, it converts the image and label to a PyTorch tensor and moves it to the GPU as needed. Next, it feeds the image into the model to obtain a prediction, and then uses the cross-entropy loss function to calculate the loss between the prediction and the true label. Finally, it sums up the losses of all the samples and divides by the number of samples to get the average loss value.
      The eval_net_BCE function is similar to eval_net, but it uses binary cross-entropy loss (nn.BCEWithLogitsLoss()) instead of cross-entropy loss. This means that it not only calculates the loss between the predicted probability and the true label, but also takes into account the log odds. This function is computed in the same way as eval_net, except that the loss function is different.

* load.py

      * This code mainly implements the following functions:
      * Defined some auxiliary functions, such as get_ids, split_ids, to_cropped_imgs, etc., which are used to handle image reading, splitting and conversion.
      * Defined ZUR_COLORMAP and ZUR_CLASSES, which represent the color mapping and classes of the ZUR dataset, respectively.
      * Defined colormap2label function to convert color mapping to label index.
      * Color2Index0 and Index2Color functions are defined to convert color labels to indexes and indexes to color labels, respectively.
      * The ten_crop function is defined for extracting multiple crop regions from a given image.
      * The create_crops and create_crops_1C functions are defined for creating crop regions from a given image.
      * The DataAug and DataAug_1C functions are defined to perform data augmentation on an image.
      * The read_images function is defined for reading images from a given directory.
      * The get_binary_imgs_and_masks function is defined for reading binarized images and masks from the specified directory.
      * The get_imgs_and_masks function is defined to read images and masks from the specified directory.

* loss.py

      This code mainly implements some loss functions in PyTorch for image classification in computer vision tasks. Below is a brief description of each loss function:
      * CrossEntropyLoss2d:2D cross entropy loss function for multi-classification problems.
      * pixCELoss2d:2D pixel-level cross entropy loss function for target segmentation tasks.
      * pixBCELoss2d:2D pixel-level binary cross-entropy loss function for target segmentation tasks.
      * weighted_BCE:Weighted binary cross-entropy loss function, used for category imbalance problem.
      * weighted_BCE_logits: weighted binary cross-entropy loss function (logit loss), used to deal with category imbalance.
      * make_one_hot: convert category index tensor to one-hot coding tensor.
      * BinaryDiceLoss:Binary Dice loss function, need to encode the input one-hot.
      * DiceLoss:Dice loss function that does not require sole-hot coding of the input.
      * VGGLoss:Calculates feature similarity loss using a pre-trained VGG model.
      * Vgg19:A class containing pre-trained VGG models that can be used to compute feature similarity loss.

* misc.py

      This code mainly implements the following functions:
      * check_mkdir(dir_name): checks and creates the directory. If the directory does not exist, it is created.
      * initialize_weights(*models):Initialize model weights. For each module of each model, initialize the weights using a normal distribution if the module is a convolutional or linear layer; if the module is a BatchNorm2D layer, initialize the weights to 1; otherwise do nothing.
      * get_upsampling_weight(in_channels, out_channels, kernel_size): calculate the upsampling weight. Calculates the weight matrix based on the given number of input channels, number of output channels and kernel size.
      * _fast_hist(label_pred, label_true, num_classes): computes the fast histogram between predictions and true values.
      * evaluate(predictions, gts, num_classes):Evaluate the accuracy between predictions and true results. Calculate confusion matrix, accuracy, category accuracy and mean square error.
      * PolyLR(optimizer, curr_iter, max_iter, lr_decay):Implements a learning rate decay strategy. During each iteration, the learning rate is adjusted according to the current iteration number and the maximum iteration number.
      * Conv2dDeformable class: implements the forward propagation of deformable convolutional layers. In the forward propagation process, the input is first cropped and filled, then the input is upsampled using bilinear interpolation, and finally a convolution kernel is applied to perform the convolution operation.

* Proportionate_datasets.py

      The main function of this code is to distribute the image files in a dataset into three different folders in the ratio of 8:1:1. (If you don't need the test set, you can change it accordingly) The steps are as follows:
      * Import the required libraries: os, shutil and random.
      * Define the source folder path (src_dir) and the three sub-folder paths (new_train, new_val, new_test).
      * Create the three subfolders (if they don't exist).
      * Get the paths of all files in the source folder and store them in the list files.
      * Calculate the number of files in each subfolder based on the ratio 8:1:1 (num_files1, num_files2, num_files3).
      * Use the random.shuffle() function to randomly shuffle the file list.
      * Copy the files into the three subfolders and distribute them according to the calculated number of files.

* rename_file.py

      The function of this code is to add a prefix to the subfolders in the specified folder. Specifically, it iterates through all the subfolders under the specified folder and renames each subfolder. The new filename will be prefixed with "20-" on top of the original filename, and at the same time, an incremental number will be added on top of the original filename. This script is mainly used for batch renaming image files.

* transform.py

      This code defines several functions whose main function is to flip, rotate and crop the image for processing image data.

* utils.py

      The provided code defines several utility functions for image processing, batch processing, evaluation metrics, and data manipulation in the context of image segmentation tasks. Let's go through the main functions:
      * get_square(img, pos): Extracts a left or right square from an image.
      * split_img_into_squares(img): Splits an image into left and right squares.
      * hwc_to_chw(img): Transposes the dimensions of an image from (H, W, C) to (C, H, W).
      * resize_and_crop(pilimg, scale=0.5, final_height=None): Resizes and crops a PIL image.
      * batch(iterable, batch_size): Yields batches from an iterable.
      * seprate_batch(dataset, batch_size): Splits a dataset into batches.
      * split_train_val(dataset, val_percent=0.05): Splits a dataset into training and validation sets.
      * normalize(x): Normalizes pixel values of an image to the range [0, 1].
      * merge_masks(img1, img2, full_w): Merges two masks horizontally.
      * rle_encode(mask_image): Run-length encoding for mask images.
      * AverageMeter: Class for computing and storing the average and current values.
      * ImageValStretch2D(img): Scales pixel values of an image.
      * ConfMap(output, pred): Computes a confidence map based on the model output and predictions.
      * accuracy(pred, label): Computes accuracy for multi-class problems.
      * align_dims(np_input, expected_dims=2): Aligns the dimensions of a NumPy array.
      * binary_accuracy(pred, label): Computes accuracy for binary classification.
      * binary_accuracy_softmax(pred, label): Computes accuracy for softmax output in binary classification.
      * intersectionAndUnion(imPred, imLab, numClass): Computes intersection and union for evaluation metrics.
      * CaclTP(imPred, imLab, numClass): Computes true positives for evaluation metrics.
      * multi_category_accuracy(pred, label, num_classes): Computes accuracy for multi-class problems with additional metrics.
      * multi_category_accuracy_ap(pred, label, num_classes): Computes accuracy for multi-class problems with additional metrics, including average precision.
      * These functions are designed to be used in the context of image segmentation and classification tasks, providing various utility and evaluation functionalities.

* voc_compute_pixel.py

      The main function of this code is to count the number of categories of all images under the VOCdevkit_path path.

      * First, import the required libraries: os, random, numpy, PIL (Python Imaging Library) and tqdm.
      * Set VOCdevkit_path to 'F:/CWLD_model/data/pred/'.
      * In the main function, first set the random number seed to 0,then get all the filenames under the VOCdevkit_path path and store them in the total_seg list.
      * Initialize an all-zero array classes_nums of length 256 to store the number of each class.
      * Use tqdm to create a progress bar that iterates through each filename in the total_seg list.
      * For each filename, splice into the full file path and check if the file exists. If it does not exist, a ValueError exception is thrown.
      * Open the picture file using the PIL library and convert it to a numpy array with an unsigned 8-bit integer data type.
      * Check if the shape of the image is greater than 2, and if so, print a warning message prompting the user to check the dataset format. Also, state that the labeled image needs to be a grayscale or eight-bit color image and that the value of each pixel point indicates the category to which the pixel point belongs.
      * Spread the image data into a one-dimensional array and use the np.bincount function to count the number of each category and accumulate the results into the classes_nums array.
      * Finally, print out each category and its corresponding number.

* voc_RGB.py
  
      The main function of this code is to count the number of blue, red, black and white pixels of all images in a folder and write the result to a text file.
      * First, import the required libraries such as os, numpy, PIL (Python Imaging Library) and skimage.
      * Set the folder path to 'E:/'.
      * Get the paths of all files ending in .png in the folder and store them in the image_paths list.
      * Initialize four counter variables for the number of blue, red, black and white pixels.
      * Iterates through each image file path in the image_paths list, opens the image using the PIL library, and adds it to the images list.
      * Prints the number of images that have been read.
      * Iterates over each image in the images list, sets the pixel values to be counted (blue, red, black, and white), and converts the image to RGB mode (if not already converted).
      * Initialize four counter variables to count the number of pixels of each color in the current image.
      * Iterate over each pixel in the image, check its color value and update the counter variables accordingly.
      * Prints the number of pixels of each color in the current image.
      * Accumulates the number of pixels of each color in the current image into the total counter variable.
      * At the end of the loop, print the total number of blue, red, black, and white pixels and their totals.
      * Write the results to a text file named 'E:/XX.txt'.

# License
Apache License Version 2.0(see LICENSE).

# Cite us
    @misc{Lei Huang2023dataset,
    title={Construction Waste Landfill Dataset of Two Districts in Beijing, China from GF-2 satellite images},
    author={Lei Huang, Shaofu Lin, Xiliang Liu},
    year={2023},
    }
