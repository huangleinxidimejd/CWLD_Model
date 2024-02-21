import os
import random
import numpy as np
import torchvision
from numpy import mean
from scipy import stats


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]


def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def seprate_batch(dataset, batch_size):
    """Yields lists by batch"""
    num_batch = len(dataset)//batch_size+1
    batch_len = batch_size
    # print (len(data))
    # print (num_batch)
    batches = []
    for i in range(num_batch):
        batches.append([dataset[j] for j in range(batch_len)])
        # print('current data index: %d' %(i*batch_size+batch_len))
        if (i+2==num_batch): batch_len = len(dataset)-(num_batch-1)*batch_size
    return(batches)


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255


def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original images) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def ImageValStretch2D(img):
    img = img*255
    #maxval = img.max(axis=0).max(axis=0)
    #minval = img.min(axis=0).min(axis=0)
    #img = (img-minval)*255/(maxval-minval)
    return img.astype(int)


def ConfMap(output, pred):
    # print(output.shape)
    n, h, w = output.shape
    conf = np.zeros(pred.shape, float)
    for h_idx in range(h):
      for w_idx in range(w):
        n_idx = int(pred[h_idx, w_idx])
        sum = 0
        for i in range(n):
          val=output[i, h_idx, w_idx]
          if val>0: sum+=val
        conf[h_idx, w_idx] = output[n_idx, h_idx, w_idx]/sum
        if conf[h_idx, w_idx]<0: conf[h_idx, w_idx]=0
    # print(conf)
    return conf


# Accuracy of multiclassification problems
def accuracy(pred, label):
    valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def align_dims(np_input, expected_dims=2):
    # Calculate the dimension of the input array
    dim_input = len(np_input.shape)
    # Initialize the dimension of the output array to the dimension of the input array
    np_output = np_input

    if dim_input>expected_dims:
        np_output = np.squeeze(np_input)
    elif dim_input<expected_dims:
        np_output = np.expand_dims(np_input, 0)
    assert len(np_output.shape) == expected_dims
    return np_output


# Calculating the accuracy of a binary classification problem
def binary_accuracy(pred, label):
    print("pred", pred.shape)
    print("lable", label.shape)
    """
    Specifically, the align_dims function is probably used to handle dimension mismatches by adjusting the dimensions to two dimensions.
    The pred and label variables go through this function and become shaped like (batch_size, num_classes).
    """
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)

    pred = (pred >= 0.5)
    label = (label >= 0.5)
    
    TP = float((pred * label).sum())
    FP = float((pred * (1-label)).sum())
    FN = float(((1-pred) * (label)).sum())
    TN = float(((1-pred) * (1-label)).sum())

    precision = TP / (TP+FP+1e-10)
    recall = TP / (TP+FN+1e-10)
    IoU = TP / (TP+FP+FN+1e-10)
    acc = (TP+TN) / (TP+FP+FN+TN)
    F1 = 0

    if acc > 0.99 and TP == 0:
        precision = 1
        recall = 1
        IoU = 1
    if precision > 0 and recall > 0:
        F1 = stats.hmean([precision, recall])

    return acc, precision, recall, F1, IoU


# Calculate the accuracy of multiclassification problems and other metrics
def multi_category_accuracy(pred, label, num_classes):
    valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)

    precisions = []
    for cls_pre in range(0, num_classes + 1):
        TP = ((pred == cls_pre) * (label == cls_pre)).sum()
        FP = ((pred == cls_pre) * (label != cls_pre)).sum()
        pre = float(TP) / (TP + FP + 1e-10)
        precisions.append(pre)
    # precision = mean(precisions)

    recalls = []
    for cls_rec in range(0, num_classes + 1):
        TP = ((pred == cls_rec) * (label == cls_rec)).sum()
        FN = ((pred != cls_rec) * (label == cls_rec)).sum()
        rec = float(TP) / (TP + FN + 1e-10)
        recalls.append(rec)
    # recall = mean(recalls)

    # F1_scores = [2 * p * r / (p + r + 1e-10) for p, r in zip(precisions, recalls)]
    # F1 = mean(F1_scores)
    F1 = 2 * precisions[3] * recalls[3] / (precisions[3] + recalls[3] + 1e-10)

    IoUs = []
    for cls_iou in range(0, num_classes + 1):
        TP = ((pred == cls_iou) * (label == cls_iou)).sum()
        FP = ((pred == cls_iou) * (label != cls_iou)).sum()
        FN = ((pred != cls_iou) * (label == cls_iou)).sum()
        iou = float(TP) / (TP + FP + FN + 1e-10)
        IoUs.append(iou)


    BERs = []
    for cls_ber in range(0, num_classes + 1):
        TP = ((pred == cls_ber) * (label == cls_ber)).sum()
        FP = ((pred == cls_ber) * (label != cls_ber)).sum()
        FN = ((pred != cls_ber) * (label == cls_ber)).sum()
        TN = ((pred != cls_ber) * (label != cls_ber)).sum()
        ber = 0.5*(float(FN) / (TP + FN + 1e-10) + float(FP) / (FP + TN + 1e-10))
        BERs.append(ber)
    return acc, precisions[3], recalls[3], F1, IoUs[3], BERs[3]


# Calculate the accuracy of multiclassification problems and other metrics
def multi_category_accuracy_ap(pred, label, num_classes):
    valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)

    precisions = []
    for cls_pre in range(0, num_classes + 1):
        TP = ((pred == cls_pre) * (label == cls_pre)).sum()
        FP = ((pred == cls_pre) * (label != cls_pre)).sum()
        pre = float(TP) / (TP + FP + 1e-10)
        precisions.append(pre)

    recalls = []
    for cls_rec in range(0, num_classes + 1):
        TP = ((pred == cls_rec) * (label == cls_rec)).sum()
        FN = ((pred != cls_rec) * (label == cls_rec)).sum()
        rec = float(TP) / (TP + FN + 1e-10)
        recalls.append(rec)

    F1 = 2 * precisions[3] * recalls[3] / (precisions[3] + recalls[3] + 1e-10)

    IoUs = []
    for cls_iou in range(0, num_classes + 1):
        TP = ((pred == cls_iou) * (label == cls_iou)).sum()
        FP = ((pred == cls_iou) * (label != cls_iou)).sum()
        FN = ((pred != cls_iou) * (label == cls_iou)).sum()
        iou = float(TP) / (TP + FP + FN + 1e-10)
        IoUs.append(iou)

    ap = []
    for cls_ap in range(0, num_classes + 1):
        mrec = np.concatenate(([0.0], recalls[cls_ap], [1.0]))
        mpre = np.concatenate(([0.0], precisions[cls_ap], [0.0]))
    return acc, precisions[3], recalls[3], F1, IoUs[3]


"""This function calculates the binary precision of the softmax output by comparing the predicted labels to the true labels"""
def binary_accuracy_softmax(pred, label):
    valid = (label < 2)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # Remove classes from unlabeled pixels in gt images.
    # We should not penalize detections in unlabeled portions of the images.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass+1))
    # print(area_intersection)

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    area_union = area_pred + area_lab - area_intersection
    # print(area_pred)
    # print(area_lab)

    return (area_intersection, area_union)


def CaclTP(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # # Remove classes from unlabeled pixels in gt images.
    # # We should not penalize detections in unlabeled portions of the images.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    TP = imPred * (imPred == imLab)
    (TP_hist, _) = np.histogram(
        TP, bins=numClass, range=(1, numClass+1))
    # print(TP.shape)
    # print(TP_hist)

    # Compute area union:
    (pred_hist, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (lab_hist, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    # print(pred_hist)
    # print(lab_hist)
    # precision = TP_hist / (lab_hist + 1e-10) + 1e-10
    # recall = TP_hist / (pred_hist + 1e-10) + 1e-10
    # # print(precision)
    # # print(recall)
    # F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    # print(F1)


    # print(area_pred)
    # print(area_lab)

    return (TP_hist, pred_hist, lab_hist)
