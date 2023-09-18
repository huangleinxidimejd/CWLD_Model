import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import functools
import sys, os


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN_res50(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True):
        super(FCN_res50, self).__init__()
        # Loading the pre-trained ResNet-50 network
        resnet = models.resnet50(pretrained=True)
        # New convolutional layer for aligning the number of input image channels into a pre-trained model for ResNet
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Copy the weights of the original convolutional layer
        newconv1.weight.data[:, :3, :, :].copy_(resnet.conv1.weight.data)
        # If the number of input channels is greater than 3,
        # the weights of the extra channels are also copied into the new convolutional layer
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])

        # Defining the layers of the network
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Changing the step size of some of the convolutional layers of ResNet's layer3 and layer4 to 1
        # is used to increase the resolution of the output feature maps
        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        # Define the convolution layer through which the output feature map passes
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False), #2048-512
                                  nn.BatchNorm2d(128, momentum=0.95),
                                  nn.ReLU())
        # Define the classifier
        self.classifier = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        # Get the size of the input feature map
        x_size = x.size()

        # forward propagation process
        x = self.layer0(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        out = self.classifier(x)

        # Return classification results and feature maps after upsampling
        return out, F.interpolate(out, x_size[2:], mode='bilinear')


class FCN_res18(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_res18, self).__init__()
        # Loading the pre-trained ResNet-18 network
        resnet = models.resnet18(pretrained)
        # New convolutional layer for aligning the number of input image channels into a pre-trained model for ResNet
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Copy the weights of the original convolutional layer
        newconv1.weight.data[:, 0:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels, :, :])
        # If the number of input channels is greater than 3,
        # the weights of the extra channels are also copied into the new convolutional layer
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])

        # Defining the layers of the network
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # Changing the step size of some of the convolutional layers of ResNet's layer3 and layer4 to 1
        # is used to increase the resolution of the output feature maps
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        # Define the convolution layer through which the output feature map passes
        self.head = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(64, momentum=0.95),
                                  nn.ReLU())
        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64, momentum=0.95),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Get the size of the input feature map
        x_size = x.size()

        # forward propagation process
        x0 = self.layer0(x)
        x = self.maxpool(x0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        x = self.classifier(x)

        # Return classification results and feature maps after upsampling
        out = F.interpolate(x, x_size[2:], mode='bilinear')
        
        return out


class FCN_res34(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_res34, self).__init__()
        # Loading the pre-trained ResNet-34 network
        resnet = models.resnet34(pretrained=False)
        # New convolutional layer for aligning the number of input image channels into a pre-trained model for ResNet
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Copy the weights of the original convolutional layer
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        # If the number of input channels is greater than 3,
        # the weights of the extra channels are also copied into the new convolutional layer
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])

        # Defining the layers of the network
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # Changing the step size of some of the convolutional layers of ResNet's layer3 and layer4 to 1
        # is used to increase the resolution of the output feature maps
        '''
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        '''
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        # Define the convolution layer through which the output feature map passes
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())

        # Define the classifier
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Get the size of the input feature map
        x_size = x.size()

        # forward propagation process
        x = self.layer0(x)  # size:1/2
        x = self.maxpool(x)  # size:1/4
        x = self.layer1(x)  # size:1/4
        x = self.layer2(x)  # size:1/8
        x = self.layer3(x)  # size:1/16
        x = self.layer4(x)
        x = self.head(x)
        out = self.classifier(x)

        # Return classification results and feature maps after upsampling
        return F.interpolate(out, x_size[2:], mode='bilinear')


class FCN_res101(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_res101, self).__init__()
        # Loading the pre-trained ResNet-101 network
        resnet = models.resnet101(pretrained)
        # A new convolutional layer is created to align the input image channels to the pre-trained model of ResNet,
        # changing the number of channels in the front and halving the image size in the back.
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Copy the weights of the original convolutional layer
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        # If the number of input channels is greater than 3,
        # the weights of the extra channels are also copied into the new convolutional layer
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        # Defining the layers of the network
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Changing the step size of some of the convolutional layers of ResNet's layer3 and layer4 to 1
        # is used to increase the resolution of the output feature maps

        # for n, m in self.layer3.named_modules():
        #     if 'conv2' in n or 'downsample.0' in n:
        #         m.stride = (1, 1)
        #
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        # Define the convolution layer through which the output feature map passes
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        # Define the classifier
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # Get the size of the input feature map
        x_size = x.size()

        # forward propagation process
        x = self.layer0(x)  # size:1/2 strid=2
        x = self.maxpool(x)  # size:1/4 strid=2

        x = self.layer1(x)  # size:1/4 strid=1

        x = self.layer2(x)  # size:1/8 strid=2
        x = self.layer3(x)  # size:1/16 strid=2
        x = self.layer4(x)  # size：1/16 strid=2

        x = self.head(x)
        out = self.classifier(x)

        # Return classification results and feature maps after upsampling
        return F.interpolate(out, x_size[2:], mode='bilinear')
