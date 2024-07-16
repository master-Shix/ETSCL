from __future__ import print_function

import pdb
import sys
import argparse
import time
import math
import random
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image, to_tensor
from main_supcon_thick import myModel1, myModel_oct
from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

class TransformAllSlices:
    def __init__(self, transforms):
        """
        初始化自定义转换。
        :param transforms: 要应用于每个切片的转换序列。
        """
        self.transforms = transforms

    def __call__(self, x):
        """
        在三维图像数据的每个切片上执行转换。
        :param x: 三维图像数据，形状为(D, H, W)。
        :return: 转换后的切片列表。
        """

        #one_time=time.time()
        x = torch.from_numpy(x)
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min)

        #print(x.shape)  #x (992,512,256)
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[2])  # 水平翻转
        else:
            x = x
        # colorjittor
        x=apply_adjustments(x,p=0.8)

        #resize
        x = x.permute(2, 0, 1)  # 现在形状是(256, 992, 512)
        x = x[:, 150:662, :]  # 现在形状是(256, 512, 512)
        #print(x.shape)
        x = x.unsqueeze(1)

        x_resized = F.interpolate(x, size=(360, 360), mode='bilinear', align_corners=False)
        x_resized = x_resized.squeeze(1)  # 移除“通道”维度，形状变为(256, 360, 360)
        x = x_resized.permute(1, 2, 0)  # 重新排列维度以匹配目标形状

        #normalize
        mean = 0.2811
        std = 0.0741

        # 对Tensor进行归一化
        x = (x - mean) / std
        # for i in range(x.shape[2]):
        #     slice = x[:, :, i]
        #     slice_pil = to_pil_image(slice)  # 将切片转换为PIL图像
        #     transformed_slice = self.transforms(slice_pil)  # 应用转换
        #
        #     transformed_slices.append(transformed_slice)
        #print(x.shape)
        #stack_slice=torch.stack(transformed_slices,dim=0)
        #final_tensor=stack_slice.squeeze(1)
        final_tensor=x.permute(2,0,1)
        #next_time = time.time()
        #final=next_time-one_time
        #print("处理的时间",final)
        return final_tensor

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=8,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='300,600,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100','path'], help='dataset')
    parser.add_argument('--classes', type=str, default='fundus',
                        choices=['fundus', 'all', 'oct'], help='dataset')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/gamma/Glaucoma_grading/training/multi-modality_images/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'path':
        # train_dataset = datasets.ImageFolder(root=opt.data_folder,
        #                                     transform=TwoCropTransform(train_transform))
        opt.n_cls = 3

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    #model = SupConResNet(name=opt.model)
    if opt.classes=="fundus":
        model = myModel1(opt)
    if opt.classes=="oct":
        model =myModel_oct(opt)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # images=torch.tensor(images)
        # images = images.cuda(non_blocking=True)
        #pdb.set_trace()
        images=images[0]
        #images = torch.cat([images[0], images[1]], dim=0)
        #images = torch.stack(images).cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        #images = [image.cuda(non_blocking=True) for image in images]
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
            pdb.set_trace()
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        #pdb.set_trace()
        acc1 = accuracy(output, labels, topk=(1,))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time_val:.3f} ({batch_time_avg:.3f})\t'
                  'DT {data_time_val:.3f} ({data_time_avg:.3f})\t'
                  'loss {loss_val:.3f} ({loss_avg:.3f})\t'
                  'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
                epoch, idx + 1, len(train_loader),
                batch_time_val=batch_time.val.item() if hasattr(batch_time.val, 'item') else batch_time.val,
                batch_time_avg=batch_time.avg.item() if hasattr(batch_time.avg, 'item') else batch_time.avg,
                data_time_val=data_time.val.item() if hasattr(data_time.val, 'item') else data_time.val,
                data_time_avg=data_time.avg.item() if hasattr(data_time.avg, 'item') else data_time.avg,
                loss_val=losses.val.item() if hasattr(losses.val, 'item') else losses.val,
                loss_avg=losses.avg.item() if hasattr(losses.avg, 'item') else losses.avg,
                top1_val=top1.val.item() if hasattr(top1.val, 'item') else top1.val,
                top1_avg=top1.avg.item() if hasattr(top1.avg, 'item') else top1.avg))
            sys.stdout.flush()
        # print info
        # if (idx + 1) % opt.print_freq == 0:
        #     pdb.set_trace()
        #     # print('Train: [{0}][{1}/{2}]\t'
        #     #       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #     #       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #     #       'loss {loss.val:.3f} ({loss.avg:.3f})\t'
        #     #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #     #        epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #     #        data_time=data_time,
        #     #         loss_val=losses.val.item() if hasattr(losses.val, 'item') else losses.val,
        #     #         loss_avg=losses.avg.item() if hasattr(losses.avg, 'item') else losses.avg,
        #     #         top1_val=top1.val.item() if hasattr(top1.val, 'item') else top1.val,
        #     #         top1_avg=top1.avg.item() if hasattr(top1.avg, 'item') else top1.avg))
        #
        #     print('Train: [{0}][{1}/{2}]\t'
        #           'BT {batch_time_val:.3f} ({batch_time_avg:.3f})\t'
        #           'DT {data_time_val:.3f} ({data_time_avg:.3f})\t'
        #           'loss {loss_val:.3f} ({loss_avg:.3f})\t'
        #           'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
        #             epoch, idx + 1, len(train_loader),
        #             batch_time_val=batch_time.val.item() if hasattr(batch_time.val, 'item') else batch_time.val,
        #             batch_time_avg=batch_time.avg.item() if hasattr(batch_time.avg, 'item') else batch_time.avg,
        #             data_time_val=data_time.val.item() if hasattr(data_time.val, 'item') else data_time.val,
        #             data_time_avg=data_time.avg.item() if hasattr(data_time.avg, 'item') else data_time.avg,
        #             loss_val=losses.val.item() if hasattr(losses.val, 'item') else losses.val,
        #             loss_avg=losses.avg.item() if hasattr(losses.avg, 'item') else losses.avg,
        #             top1_val=top1.val.item() if hasattr(top1.val, 'item') else top1.val,
        #             top1_avg=top1.avg.item() if hasattr(top1.avg, 'item') else top1.avg))
        #
        #         sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            #images = images[0]
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            #pdb.set_trace()
            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            #pdb.set_trace()
            #acc1, acc5 = accuracy(output, labels, topk=(1, 3))
            acc1 = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'loss {loss_val:.3f} ({loss_avg:.3f})\t'
                      'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
                     idx, len(val_loader),
                    loss_val=losses.val.item() if hasattr(losses.val, 'item') else losses.val,
                    loss_avg=losses.avg.item() if hasattr(losses.avg, 'item') else losses.avg,
                    top1_val=top1.val.item() if hasattr(top1.val, 'item') else top1.val,
                    top1_avg=top1.avg.item() if hasattr(top1.avg, 'item') else top1.avg))
                # print('Test: [{0}/{1}]\t'
                #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                #        idx, len(val_loader), batch_time=batch_time,
                #        loss=losses, top1=top1.val.item()))
    #pdb.set_trace()
    print(' * Test Acc1 {top1:.3f}'.format(top1=top1.avg.item()))
    # pdb.set_trace()
    # print('Test Acc_top1',top1.avg)
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion

    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        #pdb.set_trace()
       # print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2 - time1, acc=acc.item() if hasattr(acc, 'item') else acc))
        #print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2 - time1,acc.val.item()))
        print("Train epoch, accuracy",epoch,acc)
        #print(' * Test Acc@1 {top1:.3f}'.format(top1=top1.val.item()))


        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc
        torch.cuda.empty_cache()

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
