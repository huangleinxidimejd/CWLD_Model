import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch
from modelss.ResNet import FCN_res101 as FCN


BN_MOMENTUM = 0.01


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def unconv(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=2, padding=0, output_padding=0, bias=True)


def unconv1(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=4, padding=0, output_padding=0, bias=True)


class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, features, inner_features=256, out_features=512, dilations=(6, 12, 18)):
        super(ASPPModule, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))

        self.branch5_conv = nn.Conv2d(features, inner_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch5_bn = nn.BatchNorm2d(inner_features, momentum=0.95)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_features, momentum=0.95),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        # -----------------------------------------#
        #   There are five branches.
        # -----------------------------------------#
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        # -----------------------------------------#
        #   Fifth branch, global average pooling + convolution
        # -----------------------------------------#

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)

        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)

        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)

        # -----------------------------------------#
        #   Stack the five branches
        #   then 1x1 convolution integrate the features.
        # -----------------------------------------#
        feature_cat = torch.cat([feat1, feat2, feat3, feat4, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class FCN_ASPP(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_ASPP, self).__init__()
        self.FCN = FCN(in_channels, num_classes)        
        self.head = ASPPModule(2048, 64, 512)

        # self.head1 = BAM(512)

        self.low = nn.Sequential(conv1x1(256, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.low1 = nn.Sequential(conv1x1(64, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.fuse = nn.Sequential(conv3x3(64 + num_classes, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.fuse1 = nn.Sequential(conv3x3(128, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self.classifier_aux = nn.Sequential(conv1x1(512, 64), nn.BatchNorm2d(64), nn.ReLU(), conv1x1(64, num_classes))

    def forward(self, x):
        x_size = x.size()
        # Improved
        x0 = self.FCN.layer0(x)  # 1/2
        x = self.FCN.maxpool(x0)  # 1/4
        x1 = self.FCN.layer1(x)  # 1/4
        x = self.FCN.layer2(x1)  # 1/8
        x = self.FCN.layer3(x)  # 1/16
        x = self.FCN.layer4(x)

        # Enter ASPP
        x = self.head(x)

        # Encoder layer output
        aux = self.classifier_aux(x)

        x1 = self.low(x1)
        x = torch.cat((F.interpolate(aux, x1.size()[2:], mode='bilinear'), x1), 1)
        fuse = self.fuse(x)
        # out = self.classifier(fuse)

        x0 = self.low1(x0)
        x = torch.cat((F.interpolate(fuse, x0.size()[2:], mode='bilinear'), x0), 1)
        fuse = self.fuse1(x)
        out = self.classifier(fuse)

        return F.interpolate(out, x_size[2:], mode='bilinear'), F.interpolate(aux, x_size[2:], mode='bilinear')

