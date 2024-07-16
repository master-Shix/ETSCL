from __future__ import print_function
import torch.nn as nn
import pdb
import sys
import argparse
import time
import math
import random
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
from main_supcon_thick import myModel1, myModel_oct, myModel_vessel,myModel_oct3
from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy

from util import set_optimizer, save_model
from sklearn.metrics import cohen_kappa_score
from networks.resnet_big import SupConResNet, LinearClassifier,MultiLinearClassfier2,MultiLinearClassfier3
from networks.resnet_big import DSClassifier, MultiLinearClassfier3,MultiLinearClassfier4
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
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='40,80,120',
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
                        choices=['fundus', "all", 'oct','two'], help='dataset')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--ckpt_fundus', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--ckpt_oct', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--ckpt_vessel', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument("--annealing_epoch", type=int, default=10)
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


def Trans_state(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict

def set_model(opt):
    #model = SupConResNet(name=opt.model)
    if opt.classes=="fundus":
        model = myModel1(opt)
    if opt.classes=="oct":
        model =myModel_oct(opt)
    if opt.classes =="two":
        fundus_model = myModel1(opt)
        oct_model = myModel_oct(opt)

        criterion = torch.nn.CrossEntropyLoss()
        #classifier = DSClassifier(feat_dim=128, num_classes=3)
        #classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
        classifier = MultiLinearClassfier2()

        # 检测可用的设备数量
        device_count = torch.cuda.device_count()
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                devices = [torch.device(f"cuda:{i}") for i in range(device_count)]  # 列出所有可用的GPU设备
        else:
            raise NotImplementedError('This code requires GPU')

        if device_count > 1:  # 如果有多个GPU可用
            # 只用encoder
            # fundus_model.encoder = torch.nn.DataParallel(fundus_model.encoder, devices)  # 使用DataParallel将模型在多个GPU上并行计算
            # oct_model.encoder = torch.nn.DataParallel(oct_model.encoder, devices)
            # vessel_model.encoder = torch.nn.DataParallel(vessel_model.encoder, devices)

            fundus_model = torch.nn.DataParallel(fundus_model, devices)  # 使用DataParallel将模型在多个GPU上并行计算
            oct_model = torch.nn.DataParallel(oct_model, devices)
            #vessel_model = torch.nn.DataParallel(vessel_model, devices)

        ckpt_fundus = torch.load(opt.ckpt_fundus, map_location='cpu')
        state_dict_fundus = ckpt_fundus['model']
        adjusted_state_dict_fundus = {}
        adjusted_state_dict_oct = {}
        #adjusted_state_dict_vessel = {}
        for key, value in state_dict_fundus.items():
            # 将权重字典中的键名调整为匹配DataParallel封装后的模型
            new_key = key.replace('encoder.module.', 'module.encoder.')
            if not key.startswith('module.') and not key.startswith('encoder.module'):
                new_key = 'module.' + new_key
            adjusted_state_dict_fundus[new_key] = value

        ckpt_oct = torch.load(opt.ckpt_oct, map_location='cpu')
        state_dict_oct = ckpt_oct['model']
        for key, value in state_dict_oct.items():
            # 将权重字典中的键名调整为匹配DataParallel封装后的模型
            new_key = key.replace('encoder.module.', 'module.encoder.')
            if not key.startswith('module.') and not key.startswith('encoder.module'):
                new_key = 'module.' + new_key
            adjusted_state_dict_oct[new_key] = value


        # pdb.set_trace()
        fundus_model.load_state_dict(adjusted_state_dict_fundus)
        oct_model.load_state_dict(adjusted_state_dict_oct)
        #vessel_model.load_state_dict(adjusted_state_dict_vessel)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fundus_model = fundus_model.to(device)
        oct_model = oct_model.to(device)
        #vessel_model = vessel_model.to(device)
        classifier = classifier.to(device)
        cudnn.benchmark = True
        return fundus_model, oct_model, classifier, criterion
    if opt.classes =="all":
        fundus_model = myModel1(opt)
        oct_model = myModel_oct(opt)
        vessel_model = myModel_vessel(opt)
        criterion = torch.nn.CrossEntropyLoss()

        #classifier = DSClassifier(feat_dim=128,num_classes=3)
        classifier = MultiLinearClassfier3()
        #classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

        # 检测可用的设备数量
        device_count = torch.cuda.device_count()
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                devices = [torch.device(f"cuda:{i}") for i in range(device_count)]  # 列出所有可用的GPU设备
        else:
            raise NotImplementedError('This code requires GPU')

        if device_count > 1:  # 如果有多个GPU可用
            #只用encoder
            # fundus_model.encoder = torch.nn.DataParallel(fundus_model.encoder, devices)  # 使用DataParallel将模型在多个GPU上并行计算
            # oct_model.encoder = torch.nn.DataParallel(oct_model.encoder, devices)
            # vessel_model.encoder = torch.nn.DataParallel(vessel_model.encoder, devices)

            fundus_model = torch.nn.DataParallel(fundus_model, devices)  # 使用DataParallel将模型在多个GPU上并行计算
            oct_model = torch.nn.DataParallel(oct_model, devices)
            vessel_model= torch.nn.DataParallel(vessel_model, devices)

        ckpt_fundus = torch.load(opt.ckpt_fundus, map_location='cpu')
        state_dict_fundus = ckpt_fundus['model']
        adjusted_state_dict_fundus = {}
        adjusted_state_dict_oct={}
        adjusted_state_dict_vessel={}
        for key, value in state_dict_fundus.items():
            # 将权重字典中的键名调整为匹配DataParallel封装后的模型
            new_key = key.replace('encoder.module.', 'module.encoder.')
            if not key.startswith('module.') and not key.startswith('encoder.module'):
                new_key = 'module.' + new_key
            adjusted_state_dict_fundus[new_key] = value

        ckpt_oct = torch.load(opt.ckpt_oct, map_location='cpu')
        state_dict_oct = ckpt_oct['model']
        for key, value in state_dict_oct .items():
            # 将权重字典中的键名调整为匹配DataParallel封装后的模型
            new_key = key.replace('encoder.module.', 'module.encoder.')
            if not key.startswith('module.') and not key.startswith('encoder.module'):
                new_key = 'module.' + new_key
            adjusted_state_dict_oct[new_key] = value

        ckpt_vessel = torch.load(opt.ckpt_vessel, map_location='cpu')
        state_dict_vessel = ckpt_vessel['model']
        for key, value in state_dict_vessel.items():
            # 将权重字典中的键名调整为匹配DataParallel封装后的模型
            new_key = key.replace('encoder.module.', 'module.encoder.')
            if not key.startswith('module.') and not key.startswith('encoder.module'):
                new_key = 'module.' + new_key
            adjusted_state_dict_vessel[new_key] = value
        #pdb.set_trace()
        fundus_model.load_state_dict(adjusted_state_dict_fundus)
        oct_model.load_state_dict(adjusted_state_dict_oct)
        vessel_model.load_state_dict(adjusted_state_dict_vessel)



        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fundus_model = fundus_model.to(device)
        oct_model = oct_model.to(device)
        vessel_model = vessel_model.to(device)
        classifier = classifier.to(device)
        cudnn.benchmark = True
        return fundus_model, oct_model, vessel_model,classifier, criterion
    criterion = torch.nn.CrossEntropyLoss()
    #classifier = DSClassifier(feat_dim=128,num_classes=3)
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

def train_fundus(train_loader, model_fundus ,classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model_fundus.eval()

    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    kappa = AverageMeter()
    end = time.time()
    for idx, (fundus, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # images=torch.tensor(images)
        # images = images.cuda(non_blocking=True)
        #pdb.set_trace()
        fundus=fundus[0]

        fundus= fundus.cuda(non_blocking=True)

        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        #pdb.set_trace()
        # compute loss
        with torch.no_grad():

          #  oct_features = model_oct(oct)  # oct.shape 14,3,512,512
            fundus_features = model_fundus.encoder(fundus)  # fundus 14.512.128.512
          #  vessel_features = model_vessel(vessel)
            # oct_features = model_oct.encoder(oct)  #oct.shape 14,3,512,512
            # fundus_features = model_fundus.encoder(fundus)  #fundus 14.512.128.512
            # vessel_features=model_vessel.encoder(vessel)
            #pdb.set_trace()
            #combine_feature=
        output = classifier(fundus_features.detach())

       # pdb.set_trace()
        loss = criterion(output, labels)

        #loss= criterion(output_oct,labels)+criterion(output_fundus,labels)+criterion(output_vessel,labels)+criterion(output_combined,labels)
        # update metric
        losses.update(loss.item(), bsz)
        #pdb.set_trace()
        #acc1 = accuracy(output, labels, topk=(1,))
        #top1.update(acc1[0], bsz)
        #_, predicted_labels = torch.max(output, 1)
        acc1 = accuracy(output, labels, topk=(1,))
        top1.update(acc1[0], bsz)
        _, predicted_labels = torch.max(output, 1)
        if np.array_equal(labels.cpu().numpy(), predicted_labels.cpu().numpy()):
            kappa_1 = 1.0  # 完全一致
        else:
            # 如果不是全相同，正常计算Kappa系数
            kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
        #kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
        kappa.update(kappa_1,bsz)
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
                  'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'
                  ' Kappa {Kappa_val:.3f}({Kappa_avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader),
                batch_time_val=batch_time.val.item() if hasattr(batch_time.val, 'item') else batch_time.val,
                batch_time_avg=batch_time.avg.item() if hasattr(batch_time.avg, 'item') else batch_time.avg,
                data_time_val=data_time.val.item() if hasattr(data_time.val, 'item') else data_time.val,
                data_time_avg=data_time.avg.item() if hasattr(data_time.avg, 'item') else data_time.avg,
                loss_val=losses.val.item() if hasattr(losses.val, 'item') else losses.val,
                loss_avg=losses.avg.item() if hasattr(losses.avg, 'item') else losses.avg,
                top1_val=top1.val.item() if hasattr(top1.val, 'item') else top1.val,
                top1_avg=top1.avg.item() if hasattr(top1.avg, 'item') else top1.avg,
            Kappa_val=kappa.val.item() if hasattr(kappa.val, 'item') else kappa.val,
                Kappa_avg=kappa.avg.item()if hasattr(kappa.avg, 'item') else kappa.avg))

            sys.stdout.flush()


    return losses.avg, top1.avg
def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (fundus,oct, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # images=torch.tensor(images)
        # images = images.cuda(non_blocking=True)
        #pdb.set_trace()
       # images=images[0]
        fundus = fundus[0]
        oct = oct[0]
        #images = torch.cat([images[0], images[1]], dim=0)
        #images = torch.stack(images).cuda(non_blocking=True)
        fundus = fundus.cuda(non_blocking=True)
        oct = fundus.cuda(non_blocking=True)
        #images = [image.cuda(non_blocking=True) for image in images]
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        output_combined = classifier(oct_features.detach(), fundus_features.detach())
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
def train_all(train_loader, model_fundus,model_oct,model_vessel ,classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model_fundus.eval()
    model_oct.eval()
    model_vessel.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    kappa = AverageMeter()
    end = time.time()
    for idx, (oct,fundus,vessel, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # images=torch.tensor(images)
        # images = images.cuda(non_blocking=True)
        #pdb.set_trace()
        fundus=fundus[0]
        oct=oct[0]
        vessel=vessel[0]
        fundus= fundus.cuda(non_blocking=True)
        oct=oct.cuda(non_blocking=True)
        vessel = vessel.cuda(non_blocking=True)

        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        #pdb.set_trace()
        # compute loss
        with torch.no_grad():

            oct_features = model_oct(oct)  # oct.shape 14,3,512,512
            fundus_features = model_fundus(fundus)  # fundus 14.512.128.512
            vessel_features = model_vessel(vessel)
            # oct_features = model_oct.encoder(oct)  #oct.shape 14,3,512,512
            # fundus_features = model_fundus.encoder(fundus)  #fundus 14.512.128.512
            # vessel_features=model_vessel.encoder(vessel)
            #pdb.set_trace()
            #combine_feature=
        #output = classifier(features.detach())
        #output = classifier(oct_features.detach(), fundus_features.detach(),vessel_features.detach())
        #pdb.set_trace()
        output = classifier(oct_features.detach(), fundus_features.detach(),vessel_features.detach())


       # pdb.set_trace()
        #loss = criterion(output, labels)

        loss= criterion(output,labels)
        # update metric
        losses.update(loss.item(), bsz)
        #pdb.set_trace()
        #acc1 = accuracy(output, labels, topk=(1,))
        #top1.update(acc1[0], bsz)
        #_, predicted_labels = torch.max(output, 1)
        acc1 = accuracy(output, labels, topk=(1,))
        top1.update(acc1[0], bsz)
        _, predicted_labels = torch.max(output, 1)
        if np.array_equal(labels.cpu().numpy(), predicted_labels.cpu().numpy()):
            kappa_1 = 1.0  # 完全一致
        else:
            # 如果不是全相同，正常计算Kappa系数
            kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
        #kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
        kappa.update(kappa_1,bsz)
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
                  'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'
                  ' Kappa {Kappa_val:.3f}({Kappa_avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader),
                batch_time_val=batch_time.val.item() if hasattr(batch_time.val, 'item') else batch_time.val,
                batch_time_avg=batch_time.avg.item() if hasattr(batch_time.avg, 'item') else batch_time.avg,
                data_time_val=data_time.val.item() if hasattr(data_time.val, 'item') else data_time.val,
                data_time_avg=data_time.avg.item() if hasattr(data_time.avg, 'item') else data_time.avg,
                loss_val=losses.val.item() if hasattr(losses.val, 'item') else losses.val,
                loss_avg=losses.avg.item() if hasattr(losses.avg, 'item') else losses.avg,
                top1_val=top1.val.item() if hasattr(top1.val, 'item') else top1.val,
                top1_avg=top1.avg.item() if hasattr(top1.avg, 'item') else top1.avg,
            Kappa_val=kappa.val.item() if hasattr(kappa.val, 'item') else kappa.val,
                Kappa_avg=kappa.avg.item()if hasattr(kappa.avg, 'item') else kappa.avg))

            sys.stdout.flush()


    return losses.avg, top1.avg

def train_two(train_loader, model_fundus,model_oct ,classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model_fundus.eval()
    model_oct.eval()

    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    kappa = AverageMeter()
    end = time.time()
    for idx, (oct,fundus, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # images=torch.tensor(images)
        # images = images.cuda(non_blocking=True)
        #pdb.set_trace()
        fundus=fundus[0]
        oct=oct[0]

        fundus= fundus.cuda(non_blocking=True)
        oct=oct.cuda(non_blocking=True)


        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        #pdb.set_trace()
        # compute loss
        with torch.no_grad():

            oct_features = model_oct(oct)  # oct.shape 14,3,512,512
            fundus_features = model_fundus(fundus)  # fundus 14.512.128.512
            #vessel_features = model_vessel(vessel)
            #oct_features = model_oct.encoder(oct)  #oct.shape 14,3,512,512
            #fundus_features = model_fundus.encoder(fundus)  #fundus 14.512.128.512
            # vessel_features=model_vessel.encoder(vessel)
            #pdb.set_trace()
            #combine_feature=
        #output = classifier(features.detach())
        output = classifier(oct_features.detach(), fundus_features.detach())
        #output_oct, output_fundus, output_combined = classifier(oct_features.detach(),
                                                                              #
       # pdb.set_trace()
        loss = criterion(output, labels)

        #loss= criterion(output_oct,labels)+criterion(output_fundus,labels)+criterion(output_vessel,labels)+criterion(output_combined,labels)
        # update metric
        losses.update(loss.item(), bsz)
        #pdb.set_trace()
        acc1 = accuracy(output, labels, topk=(1,))
        #top1.update(acc1[0], bsz)
        _, predicted_labels = torch.max(output, 1)
        #acc1 = accuracy(output_combined, labels, topk=(1,))
        top1.update(acc1[0], bsz)
        #_, predicted_labels = torch.max(output_combined, 1)
        if np.array_equal(labels.cpu().numpy(), predicted_labels.cpu().numpy()):
            kappa_1 = 1.0  # 完全一致
        else:
            # 如果不是全相同，正常计算Kappa系数
            kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
        #kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
        kappa.update(kappa_1,bsz)
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
                  'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'
                  ' Kappa {Kappa_val:.3f}({Kappa_avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader),
                batch_time_val=batch_time.val.item() if hasattr(batch_time.val, 'item') else batch_time.val,
                batch_time_avg=batch_time.avg.item() if hasattr(batch_time.avg, 'item') else batch_time.avg,
                data_time_val=data_time.val.item() if hasattr(data_time.val, 'item') else data_time.val,
                data_time_avg=data_time.avg.item() if hasattr(data_time.avg, 'item') else data_time.avg,
                loss_val=losses.val.item() if hasattr(losses.val, 'item') else losses.val,
                loss_avg=losses.avg.item() if hasattr(losses.avg, 'item') else losses.avg,
                top1_val=top1.val.item() if hasattr(top1.val, 'item') else top1.val,
                top1_avg=top1.avg.item() if hasattr(top1.avg, 'item') else top1.avg,
            Kappa_val=kappa.val.item() if hasattr(kappa.val, 'item') else kappa.val,
                Kappa_avg=kappa.avg.item()if hasattr(kappa.avg, 'item') else kappa.avg))

            sys.stdout.flush()


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
def validate_fundus(val_loader, model_fundus,classifier, criterion, opt):
    """validation"""
    model_fundus.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    kappa = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for idx, (fundus,labels) in enumerate(val_loader):
            #images = images[0]

            # fundus = fundus.cuda(non_blocking=True)
            # oct = oct.cuda(non_blocking=True)
            fundus = fundus.float().cuda()
            #oct=oct.float().cuda()
            #vessel=vessel.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            #pdb.set_trace()
            # forward
            output = classifier(model_fundus.encoder(fundus))
            #pdb.set_trace()
            #output = classifier(model_oct.encoder(oct), model_fundus.encoder(fundus),model_vessel.encoder(vessel))
            #output = classifier(model_oct(oct), model_fundus(fundus), model_vessel(vessel))

            #oct_features = model_oct(oct)  # oct.shape 14,3,512,512
            fundus_features = model_fundus.encoder(fundus.detach())  # fundus 14.512.128.512
            #vessel_features = model_vessel(vessel)
            loss = criterion(fundus_features, labels)
            # loss = criterion(output_oct, labels) + criterion(output_fundus, labels) + criterion(output_vessel,
            #                                                                                     labels) + criterion(
            #     output_combined, labels)
            # update metric
            losses.update(loss.item(), bsz)
            #loss = criterion(output, labels)
            acc1 = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)
            _, predicted_labels = torch.max(output, 1)
            #kappa = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
            #pdb.set_trace()
            if np.array_equal(labels.cpu().numpy(), predicted_labels.cpu().numpy()):
                kappa_1 = 1.0  # 完全一致
            else:
                # 如果不是全相同，正常计算Kappa系数
                kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())

            #kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
            kappa.update(kappa_1,bsz)
            # update metric
            losses.update(loss.item(), bsz)
            #pdb.set_trace()
            #acc1, acc5 = accuracy(output, labels, topk=(1, 3))
           # acc1 = accuracy(output_combined, labels, topk=(1,))
           # top1.update(acc1[0], bsz)

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
    #print(' * Test Acc1 {top1:.3f}'.format(top1=top1.avg.item()))
    print(' * Test Kappa {kappa:.3f}'.format(kappa=kappa.avg.item()))
    print('Test: ' 'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'
          ' Kappa {Kappa_val:.3f}({Kappa_avg:.3f})\t'.format(
        top1_val=top1.val.item() if hasattr(top1.val, 'item') else top1.val,
        top1_avg=top1.avg.item() if hasattr(top1.avg, 'item') else top1.avg,
        Kappa_val=kappa.val.item() if hasattr(kappa.val, 'item') else kappa.val,
        Kappa_avg=kappa.avg.item() if hasattr(kappa.avg, 'item') else kappa.avg))
    # pdb.set_trace()
    # print('Test Acc_top1',top1.avg)
    return losses.avg, top1.avg
def validate_all(val_loader, model_fundus,model_oct, model_vessel,classifier, criterion, opt):
    """validation"""
    model_fundus.eval()
    model_oct.eval()
    model_vessel.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    kappa = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for idx, (oct,fundus, vessel,labels) in enumerate(val_loader):
            #images = images[0]

            # fundus = fundus.cuda(non_blocking=True)
            # oct = oct.cuda(non_blocking=True)
            fundus = fundus.float().cuda()
            oct=oct.float().cuda()
            vessel=vessel.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            #pdb.set_trace()
            # forward
          #  output = classifier(model.encoder(images))
            #pdb.set_trace()
            #output = classifier(model_oct.encoder(oct), model_fundus.encoder(fundus),model_vessel.encoder(vessel))
            #output = classifier(model_oct(oct), model_fundus(fundus), model_vessel(vessel))
            oct_features = model_oct(oct)  # oct.shape 14,3,512,512
            fundus_features = model_fundus(fundus)  # fundus 14.512.128.512
            vessel_features = model_vessel(vessel)
            output= classifier(oct_features.detach(),fundus_features.detach(), vessel_features.detach())
            loss = criterion(output, labels)
            # update metric
            #losses.update(loss.item(), bsz)
            #loss = criterion(output, labels)
            acc1 = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)
            _, predicted_labels = torch.max(output, 1)
            #kappa = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
            #pdb.set_trace()
            if np.array_equal(labels.cpu().numpy(), predicted_labels.cpu().numpy()):
                kappa_1 = 1.0  # 完全一致
            else:
                # 如果不是全相同，正常计算Kappa系数
                kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())

            #kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
            kappa.update(kappa_1,bsz)
            # update metric
            losses.update(loss.item(), bsz)
            #pdb.set_trace()
            #acc1, acc5 = accuracy(output, labels, topk=(1, 3))
           # acc1 = accuracy(output_combined, labels, topk=(1,))
           # top1.update(acc1[0], bsz)

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
    #print(' * Test Acc1 {top1:.3f}'.format(top1=top1.avg.item()))
    print(' * Test Kappa {kappa:.3f}'.format(kappa=kappa.avg.item()))
    print('Test: ' 'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'
          ' Kappa {Kappa_val:.3f}({Kappa_avg:.3f})\t'.format(
        top1_val=top1.val.item() if hasattr(top1.val, 'item') else top1.val,
        top1_avg=top1.avg.item() if hasattr(top1.avg, 'item') else top1.avg,
        Kappa_val=kappa.val.item() if hasattr(kappa.val, 'item') else kappa.val,
        Kappa_avg=kappa.avg.item() if hasattr(kappa.avg, 'item') else kappa.avg))
    # pdb.set_trace()
    # print('Test Acc_top1',top1.avg)
    return losses.avg, top1.avg
def validate_two(val_loader, model_fundus,model_oct,classifier, criterion, opt):
    """validation"""
    model_fundus.eval()
    model_oct.eval()

    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    kappa = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for idx, (oct,fundus, labels) in enumerate(val_loader):
            #images = images[0]

            # fundus = fundus.cuda(non_blocking=True)
            # oct = oct.cuda(non_blocking=True)
            fundus = fundus.float().cuda()
            oct=oct.float().cuda()

            labels = labels.cuda()
            bsz = labels.shape[0]
            #pdb.set_trace()
            # forward
          #  output = classifier(model.encoder(images))
            #pdb.set_trace()
           # output = classifier(model_oct.encoder(oct), model_fundus.encoder(fundus))

            oct_features = model_oct(oct)  # oct.shape 14,3,512,512
            fundus_features = model_fundus(fundus)  # fundus 14.512.128.512
            #vessel_features = model_vessel(vessel)
            #output_oct, output_fundus, output_combined = classifier(oct_features.detach(),
            #                                                                       fundus_features.detach(),
            #                                                                      )
            #loss = criterion(output_oct, labels) + criterion(output_fundus, labels) + criterion(output_vessel,
            #                                                                                    labels) + criterion(
            #    output_combined, labels)
            # update metric
            output = classifier(oct_features.detach(), fundus_features.detach())
            loss = criterion(output, labels)
            losses.update(loss.item(), bsz)
            output = classifier(oct_features.detach(), fundus_features.detach())
            loss = criterion(output, labels)
            acc1 = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)
            _, predicted_labels = torch.max(output, 1)
            #kappa = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
            #pdb.set_trace()
            if np.array_equal(labels.cpu().numpy(), predicted_labels.cpu().numpy()):
                kappa_1 = 1.0  # 完全一致
            else:
                # 如果不是全相同，正常计算Kappa系数
                kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())

            #kappa_1 = cohen_kappa_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
            kappa.update(kappa_1,bsz)
            # update metric
            losses.update(loss.item(), bsz)
            #pdb.set_trace()
            #acc1, acc5 = accuracy(output, labels, topk=(1, 3))
           # acc1 = accuracy(output_combined, labels, topk=(1,))
           # top1.update(acc1[0], bsz)

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
    #print(' * Test Acc1 {top1:.3f}'.format(top1=top1.avg.item()))
    print(' * Test Kappa {kappa:.3f}'.format(kappa=kappa.avg.item()))
    print('Test: ' 'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'
          ' Kappa {Kappa_val:.3f}({Kappa_avg:.3f})\t'.format(
        top1_val=top1.val.item() if hasattr(top1.val, 'item') else top1.val,
        top1_avg=top1.avg.item() if hasattr(top1.avg, 'item') else top1.avg,
        Kappa_val=kappa.val.item() if hasattr(kappa.val, 'item') else kappa.val,
        Kappa_avg=kappa.avg.item() if hasattr(kappa.avg, 'item') else kappa.avg))
    # pdb.set_trace()
    # print('Test Acc_top1',top1.avg)
    return losses.avg, top1.avg
def main():
    best_acc = 0
    opt = parse_option()

    # build data loader

    train_loader, val_loader = set_loader(opt)

    # build model and criterion
   # pdb.set_trace()
    if opt.classes=="all":
        model_fundus,model_oct,model_vessel, classifier, criterion = set_model(opt)
    elif opt.classes=="two":
        model_fundus, model_oct, classifier, criterion = set_model(opt)
    else:
        model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        if opt.classes == "all":
            loss, acc = train_all(train_loader, model_fundus,model_oct,model_vessel ,classifier, criterion,
                              optimizer, epoch, opt)
        elif opt.classes == "two":
            loss, acc = train_two(train_loader, model_fundus,model_oct ,classifier, criterion,
                              optimizer, epoch, opt)
        elif opt.classes == "fundus":
            loss, acc = train_fundus(train_loader, model, classifier, criterion,
                              optimizer, epoch, opt)
        else:
            loss, acc = train(train_loader, model, classifier, criterion,
                              optimizer, epoch, opt)
        time2 = time.time()
        #pdb.set_trace()
       # print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2 - time1, acc=acc.item() if hasattr(acc, 'item') else acc))
        #print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2 - time1,acc.val.item()))
        print("Train epoch, accuracy",epoch,acc)
        #print(' * Test Acc@1 {top1:.3f}'.format(top1=top1.val.item()))


        # eval for one epoch
        if opt.classes == "all":

            loss, val_acc = validate_all(val_loader, model_fundus,model_oct, model_vessel,classifier, criterion, opt)
        elif opt.classes == "fundus":
            loss, val_acc = validate_fundus(val_loader, model, classifier, criterion, opt)
        elif opt.classes == "two":
            loss, val_acc = validate_two(val_loader, model_fundus,model_oct,classifier, criterion, opt)
        else:
            loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        #pdb.set_trace()
        if val_acc > best_acc:
            best_acc = val_acc
            # save the last model
            opt.save_folder      ='/root/autodl-tmp/SupContrast/save/SupCon/best_mode/3_scl_softmax'
            save_file_fundus     = os.path.join(opt.save_folder, 'fundus',str(best_acc)+'last.pth')
            save_file_oct        = os.path.join(opt.save_folder, 'oct', str(best_acc)+'last.pth')
            save_file_vessel     = os.path.join(opt.save_folder, 'vessel', str(best_acc)+'last.pth')
            save_file_classifier = os.path.join(opt.save_folder, 'classifier', str(best_acc)+'last.pth')


            save_model(model_fundus, optimizer, opt, opt.epochs, save_file_fundus)
            save_model(model_oct,    optimizer, opt, opt.epochs, save_file_oct)
            save_model(model_vessel, optimizer, opt, opt.epochs, save_file_vessel)
            save_model(classifier,   optimizer, opt, opt.epochs, save_file_classifier)
        torch.cuda.empty_cache()

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
