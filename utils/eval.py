import torch
import torch.nn.functional as F
import torch.nn as nn
from dice_loss import dice_coeff

"""eval_net(net, dataset, gpu=True): used to evaluate the performance of the network on the dataset. Calculates the 
cross-entropy loss for each sample and finally returns the average loss. Where the input dataset is an iterator that 
returns a tuple of the form (images, label) for each iteration, where image is the input image and label is the 
corresponding real label. eval_net_BCE(net, dataset, gpu=True): similar to eval_net, but using binary cross-entropy 
loss. Where again, the input dataset is an iterator, and each iteration returns a tuple of the form (images, label), 
where image is the input image and label is the corresponding real label. These functions receive a net object, i.e., 
a model instance, which can be used after training to evaluate the model's performance on new data. If gpu=True, 
the GPU is used to compute the"""


def eval_net(net, dataset, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    n=len(dataset)
    for i, b in enumerate(dataset):
        # img = b[0]
        # true_mask = b[1]
        # img = torch.from_numpy(img).unsqueeze(0)
        # true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        #
        # if gpu:
        #     img = img.cuda()
        #     true_mask = true_mask.cuda()

        img = torch.from_numpy(b[0]).unsqueeze(0).float()
        label = torch.from_numpy(b[1]).unsqueeze(0).long()
        if gpu:
            img = img.cuda()
            label = label.cuda()

        pred = net(img)
        loss = nn.CrossEntropyLoss()
        loss = loss(pred, label)

        tot += loss.item()
    return tot / n

def eval_net_BCE(net, dataset, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    n=len(dataset)
    for i, b in enumerate(dataset):
        # img = b[0]
        # true_mask = b[1]
        # img = torch.from_numpy(img).unsqueeze(0)
        # true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        #
        # if gpu:
        #     img = img.cuda()
        #     true_mask = true_mask.cuda()

        img = torch.from_numpy(b[0]).unsqueeze(0).float()
        label = torch.from_numpy(b[1]).unsqueeze(0).float()
        if gpu:
            img = img.cuda()
            label = label.cuda()

        pred = net(img)
        pred_flat = pred.view(-1)
        labels_flat = label.view(-1)
        loss = nn.BCEWithLogitsLoss()
        loss = loss(pred_flat, labels_flat)

        tot += loss.item()
    return tot / n
