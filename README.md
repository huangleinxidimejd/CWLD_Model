# CWLD_Model
* DeepLabV3+ network is a multi-scale, multipath parallel convolutional neural network proposed by Chen,L.C et al. in 2018.The main advantage of DeepLabV3+ network is its excellent image segmentation performance and robustness. Its core innovation is the introduction of an encoding-decoding structure that combines low-level semantic information with high-level semantic information, thus improving the segmentation accuracy of the network. We have improved the DeepLabV3+ network by dividing it into four main components: the backbone network, the Atrous Spatial Pyramid Pooling (ASPP) module, the encoding structure, and the decoding structure. The backbone network uses ResNet-101 with a residual structure for feature extraction from the original image, and the ASPP module is a null convolution with different expansion rates for fusion of features at different scales, which increases the sensory field without losing image information. The coding structure is used for feature extraction while reducing the size of the feature map to reduce the computational complexity. The decoding structure is used for up-sampling to recover the spatial detail information of the image, and for fusion of deep and shallow features to obtain finer recognition results.
![网络模型](https://github.com/huangleinxidimejd/CWLD_Model/assets/42790126/228cbb85-ab7b-4330-a600-7dec89192857)

## Data
The dataset used to train the model is the CWLD dataset published on Zenodo.[]

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
# License
Apache License Version 2.0(see LICENSE).

# Cite us
  @misc{Lei Huang2023dataset,
  title={Construction Waste Landfill Dataset of Two Districts in Beijing, China from GF-2 satellite images},
  author={Lei Huang, Shaofu Lin, Xiliang Liu},
  year={2023},
}
