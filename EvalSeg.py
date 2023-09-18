import math
import os
import time
import torch
import numpy as np
import torch.autograd
import torchvision
from matplotlib import pyplot as plt
from skimage import io
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn

from modelss.Improved_DeeplabV3_plus import FCN_ASPP as Net
# from modelss.PSPNet import Pspnet as Net
# from modelss.SegNet import SegNet as Net
# from modelss.Unet import UNet as Net
from dataset import Waste as RS
from utils.loss import CrossEntropyLoss2d
# from utils.utils import multi_category_accuracy as accuracy
from utils.utils import multi_category_accuracy as accuracy
from utils.utils import intersectionAndUnion, AverageMeter, CaclTP

#################################
DATA_NAME = 'WasteSeg'
#################################

working_path = os.path.dirname(os.path.abspath(__file__))

args = {
    'gpu': True,
    'batch_size': 1,
    'net_name': 'Improved_DeeplabV3_plus',
    'load_path': os.path.join(working_path, 'checkpoints', 'WasteSeg', 'Improved_DeeplabV3_plus_192e_OA96.21_F88.89_IoU82.08.pth')
}



def soft_argmax(seg_map):
    assert seg_map.dim() == 4
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1000.0 
    b,c,h,w, = seg_map.shape
    print(seg_map.shape)
    soft_max = F.softmax(seg_map*alpha, dim=1)
    return soft_max


def main():
    net = Net(num_classes=RS.num_classes + 1)

    net.load_state_dict(torch.load(args['load_path']), strict=False)  # strict = False
    # net.upsample2 = Identity()
    net = net.cuda()
    # It sets the model's mode of operation to evaluation mode
    net.eval()
    print('Model loaded.')
    pred_path = os.path.join(RS.root, 'pred', args['net_name'])
    if not os.path.exists(pred_path): os.makedirs(pred_path)
    info_txt_path = os.path.join(pred_path, 'info.txt')
    f = open(info_txt_path, 'w+')

    #test_set = RS.RS('test')
    pred_name_list = RS.get_file_name('val')
    test_set = RS.RS('val')
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], num_workers=4, shuffle=False)
    predict(net, test_loader, pred_path, pred_name_list, f)


def predict(net, pred_loader, pred_path, pred_name_list, f_out=None):
    output_info = f_out is not None

    """
    acc_meter, precision_meter, recall_meter, F1_meter, and IoU_meter are all instances of the custom AverageMeter class.
    """
    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    BER_meter = AverageMeter()

    acc_list = []
    precision_list = []
    recall_list = []
    F1_list = []
    IoU_list = []
    BER_list = []

    total_iter = len(pred_loader)
    num_files = len(pred_name_list)

    crop_nums = math.ceil(total_iter / num_files)

    for vi, data in enumerate(pred_loader):
        imgs, labels = data
        # imgs = data
        imgs = imgs.cuda().float()
        with torch.no_grad(): 
            # outputs, _ = net(imgs)
            outputs = net(imgs)
            # print("outputs",imgs.shape)
            # outputs = F.sigmoid(outputs)
            print(outputs.shape)
        output = outputs.detach().cpu()
        output = torch.argmax(output, dim=1)
        outputs = output.numpy()


        for i in range(args['batch_size']):
            idx = vi*args['batch_size'] + i
            file_idx = int(idx/crop_nums)
            crop_idx = idx % crop_nums
            if (idx>=total_iter): break
            pred = outputs[i]
            label = labels[i].detach().cpu().numpy()

            acc, precision, recall, F1, IoU, BER = accuracy(pred, label, RS.num_classes)

            acc_meter.update(acc)
            precision_meter.update(precision)
            recall_meter.update(recall)
            F1_meter.update(F1)
            IoU_meter.update(IoU)
            BER_meter.update(BER)

            # Record precisions, recalls into lists
            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            F1_list.append(F1)
            IoU_list.append(IoU)
            BER_list.append(BER)

            """
            The predicted labels are converted to color images using the RS.Index2Color function, the
            and saved to disk using the imsave function in the skimage.io module.
            The filename of the saved image is constructed using pred_name_list and crop_idx.
            """
            print(pred.shape)
            pred_color = RS.Index2Color(pred.squeeze())
            if crop_nums > 1: pred_name = os.path.join(pred_path, pred_name_list[file_idx]+'_%d.png'%crop_idx)
            else: pred_name = os.path.join(pred_path, pred_name_list[file_idx]+'.png')
            io.imsave(pred_name, pred_color)

            print('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f' % (idx, total_iter, acc*100, precision*100, recall*100, F1*100, IoU*100))
            if output_info:
                f_out.write('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f\n' % (idx, total_iter, acc*100, precision*100, recall*100, F1*100, IoU*100))

    if output_info:
        f_out.write('precision %s\n' % precision_list)
        f_out.write('recall %s\n' % recall_list)
        f_out.write('acc %s\n' % acc_list)
        f_out.write('F1 %s\n' % F1_list)
        f_out.write('IoU %s\n' % IoU_list)
        f_out.write('BER %s\n' % BER_list)

    # Visualization
    # Plotting curves
    x_1 = list(range(1, len(pred_loader) + 1))
    # plot the lines separately with their own settings and label
    plt.plot(x_1, precision_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("precision.png")
    plt.clf()

    x_2 = list(range(1, len(pred_loader) + 1))
    plt.plot(x_2, recall_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("recall.png")
    plt.clf()

    x_3 = list(range(1, len(pred_loader) + 1))
    plt.plot(x_3, acc_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("acc.png")
    plt.clf()

    x_4 = list(range(1, len(pred_loader) + 1))
    plt.plot(x_4, F1_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("F1.png")
    plt.clf()

    x_5 = list(range(1, len(pred_loader) + 1))
    plt.plot(x_5, IoU_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("IoU.png")
    plt.clf()

    x_6 = list(range(1, len(pred_loader) + 1))
    plt.plot(x_6, BER_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("BER.png")
    plt.clf()

    print('avg Acc %.2f, Pre %.2f, Recall %.2f, F1 %.2f, IOU %.2f, BER %.2f\n' % (acc_meter.avg*100, precision_meter.avg*100, recall_meter.avg*100, F1_meter.avg*100, IoU_meter.avg*100, BER_meter.avg*100))

    if output_info:
        f_out.write('Acc %.2f\n' % (acc_meter.avg*100))
        f_out.write('Avg Precision %.2f\n' % (precision_meter.avg*100))
        f_out.write('Avg Recall %.2f\n' % (recall_meter.avg*100))
        f_out.write('Avg F1 %.2f\n' % (F1_meter.avg*100))
        f_out.write('mIoU %.2f\n' % (IoU_meter.avg*100))
        f_out.write('mBER %.2f\n' % (BER_meter.avg*100))
    return F1_meter.avg


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    main()
