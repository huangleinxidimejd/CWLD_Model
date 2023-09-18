import os
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch import optim
import torch.autograd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from skimage import io
from dataset import Waste as RS
# from modelss.Improved_DeeplabV3_plus import FCN_ASPP as Net
# from modelss.Unet import UNet as Net
# from modelss.SegNet import SegNet as Net
from modelss.PSPNet import Pspnet as Net
# from modelss.DAMM_DeepLabv3_plus_162 import DAMM_DeepLabv3_plus_162 as Net
# from torch.nn.modules.loss import CrossEntropyLoss as CEloss
from utils.utils import multi_category_accuracy as accuracy
from utils.utils import intersectionAndUnion, AverageMeter
# from modelss.DiResSeg import DiResSeg as Net

NET_NAME = 'PSPNet'
DATA_NAME = 'WasteSeg'
working_path = os.path.abspath('.')

args = {
    'train_batch_size': 4,
    'val_batch_size': 4,
    'train_crop_size': 512,
    'val_crop_size': 512,
    'lr': 0.001,  # The learning rate used during training. It determines how fast the model learns from the data
    'epochs': 200,
    'gpu': True,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 100,
    'predict_step': 5,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),  # Catalogs where model predictions are kept
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),  # Directory where model checkpoints are kept
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME),  # Directory where model logs are kept
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'xx.pth')
}

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
writer = SummaryWriter(args['log_dir'])


def main():
    net = Net(num_classes=RS.num_classes + 1).cuda()

    train_set = RS.RS('train', random_flip=True)
    val_set = RS.RS('val')

    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)

    # weight = torch.tensor(1.0)
    # criterion = torch.nn.BCEWithLogitsLoss().cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()

    # Define the optimizer as SGD while tuning the learning rate using the StepLR learning rate scheduler.
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)

    # optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=args['weight_decay'])

    train(train_loader, net, criterion, optimizer, scheduler, 0, args, val_loader)
    writer.close()
    print('Training finished.')
    # predict(net, args)


def train(train_loader, net, criterion, optimizer, scheduler, curr_epoch, train_args, val_loader):
    # Visual presentation
    Acc = []
    train_F1 = []
    val_F1 = []
    train_loss = []
    val_loss = []
    cu_ep = []

    bestaccT = 0
    bestF = 0
    bestIoU = 0
    bestloss = 1
    begin_time = time.time()
    # Total number of iterations, used to adjust the learning rate
    all_iters = float(len(train_loader) * args['epochs'])
    # fgm = FGM(net)
    # criterionVGG = VGGLoss()
    while True:
        # Clear GPU Cache
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        F1_meter = AverageMeter()
        train_main_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_learning_rate(optimizer, running_iter, all_iters, args)
            imgs, labels = data
            if args['gpu']:
                imgs = imgs.cuda().float()
                labels = labels.cuda().long()

            # labels_s = F.interpolate(label, scale_factor=1/8, mode='area')

            optimizer.zero_grad()
            outputs, aux = net(imgs)
            # outputs = net(imgs)

            assert outputs.shape[1] == RS.num_classes + 1

            loss_main = criterion(outputs, labels)
            loss_aux = criterion(aux, labels)
            loss = loss_main + loss_aux * 0.3

            # loss = criterion(outputs, labels)
            # out_sigmoid = F.sigmoid(outputs)

            # loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach()

            _, preds = torch.max(outputs, dim=1)
            preds = preds.numpy()

            # preds = out_sigmoid.cpu().detach().numpy()
            # batch_valid_sum = 0

            F1_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels):
                acc, precision, recall, F1, IoU, BER = accuracy(pred, label, RS.num_classes)
                if F1 > 0:
                    F1_curr_meter.update(F1)
            if F1_curr_meter.avg is not None:
                F1_meter.update(F1_curr_meter.avg)
            else:
                F1_meter.update(0)
            train_main_loss.update(loss.cpu().detach().numpy())

            # train_aux_loss.update(aux_loss, batch_pixel_sum)

            curr_time = time.time() - start

            if (i + 1) % train_args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f F1 %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_main_loss.val, F1_meter.avg * 100))
                writer.add_scalar('train loss', train_main_loss.val, running_iter)
                rex_loss = train_main_loss.val
                writer.add_scalar('train F1', F1_meter.avg, running_iter)
                # writer.add_scalar('train_aux_loss', train_aux_loss.avg, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        val_F, val_acc, val_IoU, loss_v = validate(val_loader, net, criterion)

        Acc.append(val_acc)
        val_F1.append(val_F)
        val_loss.append(loss_v)
        train_loss.append(train_main_loss.avg)
        train_F1.append(F1_meter.avg)
        cu_ep.append(curr_epoch)

        if val_F > bestF:
            bestF = val_F
            bestloss = loss_v
            bestIoU = val_IoU
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME + '_%de_OA%.2f_F%.2f_IoU%.2f.pth' % (
                curr_epoch, val_acc * 100, val_F * 100, val_IoU * 100)))
        print('Total time: %.1fs Best rec: Val %.2f, Val_loss %.4f BestIOU: %.2f' % (
            time.time() - begin_time, bestF * 100, bestloss, bestIoU * 100))
        curr_epoch += 1

        # scheduler.step()
        # Visualization of ACC, F1, LOSS images
        if curr_epoch >= train_args['epochs']:
            print('Acc', Acc)
            print('train_F1', train_F1)
            print('val_F1', val_F1)
            print('train_loss', train_loss)
            print('val_loss', val_loss)

            with open('train_all_eval.txt', 'a') as file0:
                print('Acc:%s\n' % Acc,
                      'train_F1:%s\n' % train_F1,
                      'val_F1:%s\n' % val_F1,
                      'train_loss:%s\n' % train_loss,
                      'val_loss:%s\n' % val_loss, file=file0)

            # Plotting 5 curve curves
            x_all = cu_ep
            # create four lists of (x, y) pairs using zip()
            line1 = list(zip(x_all, val_loss))
            line2 = list(zip(x_all, train_loss))
            line3 = list(zip(x_all, Acc))
            line4 = list(zip(x_all, val_F1))
            line5 = list(zip(x_all, train_F1))
            # plot the lines separately with their own settings and label
            plt.plot([p[0] for p in line1], [p[1] for p in line1], color='green', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='val_loss')
            plt.plot([p[0] for p in line2], [p[1] for p in line2], color='red', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='train_loss')
            plt.plot([p[0] for p in line3], [p[1] for p in line3], color='blue', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='Acc')
            plt.plot([p[0] for p in line4], [p[1] for p in line4], color='black', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='val_F1')
            plt.plot([p[0] for p in line5], [p[1] for p in line5], color='yellow', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='train_F1')
            # add a legend to the plot
            plt.legend()
            # plt.show()
            plt.savefig("train_all_eval.png")
            plt.clf()
            return


def validate(val_loader, model_seg, criterion, save_pred=True):
    # the following code is written assuming that batch size is 1
    model_seg.eval()
    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()

    for vi, data in enumerate(val_loader):
        imgs, labels = data

        if args['gpu']:
            imgs = imgs.cuda().float()
            labels = labels.cuda().long()

        with torch.no_grad():
            out, aux = model_seg(imgs)
            # out = model_seg(imgs)
            loss = criterion(out, labels)

            # out_bn = F.sigmoid(out)  # soft_argmax(out)[:,1,:,:]
        val_loss.update(loss.cpu().detach().numpy())

        out = out.cpu().detach()
        labels = labels.cpu().detach().numpy()

        _, preds = torch.max(out, dim=1)
        preds = preds.numpy()

        for (pred, label) in zip(preds, labels):
            acc, precision, recall, F1, IoU, BER = accuracy(pred, label, RS.num_classes)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)
        if save_pred and vi == 0:
            pred_color = RS.Index2Color(preds[0].squeeze())
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '.png'), pred_color)
            print('Prediction saved!')

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f, F1: %.2f, Accuracy: %.2f' % (
        curr_time, val_loss.average(), F1_meter.avg * 100, Acc_meter.average() * 100))

    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_loss.avg


def adjust_learning_rate(optimizer, curr_iter, all_iter, args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 1.5)
    running_lr = args['lr'] * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    start = time.time()
    main()
