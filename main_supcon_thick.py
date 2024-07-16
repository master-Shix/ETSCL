from __future__ import print_function
import random
import pdb
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
import sys
import argparse
import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import timm
from torch.utils.data import Dataset
import cv2

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
import apex
from apex import amp, optimizers
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
        #x = x[::2, :, :]  # 现在形状是(128, 512, 512)
        x = x[::2, :, :]  # 现在形状是(128, 512, 512)
        # print("qian",x.shape)
        # x = x.unsqueeze(1)
        # print("hou",x.shape)
        #x_resized = F.interpolate(x, size=(360, 360), mode='bilinear', align_corners=False)
        # x_resized = x_resized.squeeze(1)  # 移除“通道”维度，形状变为(256, 360, 360)
        # x = x_resized.permute(1, 2, 0)  # 重新排列维度以匹配目标形状

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
        #final_tensor=x.permute(2,0,1)
        #next_time = time.time()
        #final=next_time-one_time
        #print("处理的时间",final)
        return x

class TransformAllSlices3:
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
        #x = x[::2, :, :]  # 现在形状是(128, 512, 512)
        x = x[125:128, :, :]  # 现在形状是(3, 512, 512)
        # print("qian",x.shape)
        # x = x.unsqueeze(1)
        # print("hou",x.shape)
        #x_resized = F.interpolate(x, size=(360, 360), mode='bilinear', align_corners=False)
        # x_resized = x_resized.squeeze(1)  # 移除“通道”维度，形状变为(256, 360, 360)
        # x = x_resized.permute(1, 2, 0)  # 重新排列维度以匹配目标形状

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
        #final_tensor=x.permute(2,0,1)
        #next_time = time.time()
        #final=next_time-one_time
        #print("处理的时间",final)
        return x
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
class ResNet_255(nn.Module):
    def __init__(self, block, num_blocks, in_channel=256, zero_init_residual=False):
        super(ResNet_255, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

class ResNet_128(nn.Module):
    def __init__(self, block, num_blocks, in_channel=128, zero_init_residual=False):
        super(ResNet_128, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
class ResNet_1(nn.Module):
    def __init__(self, block, num_blocks, in_channel=1, zero_init_residual=False):
        super(ResNet_1, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet50_255(**kwargs):
    return ResNet_255(Bottleneck, [3, 4, 6, 3], **kwargs)
def resnet50_128(**kwargs):
    return ResNet_128(Bottleneck, [3, 4, 6, 3], **kwargs)
def resnet50_1(**kwargs):
    return ResNet_1(Bottleneck, [3, 4, 6, 3], **kwargs)
def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='2,4,6,8,10,15',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--classes', type=str, default='fundus',
                        choices=['oct', 'fundus', 'vessel',"all_data"], help='what class data will be need')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR','NTXent'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument("--annealing_epoch", type=int, default=10)

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        opt.data_folder = './root/autodl-tmp/SupContrast/datasets/gamma/Glaucoma_grading/'

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './root/autodl-tmp/SupContrast/datasets/gamma/Glaucoma_grading/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_0922_thick384_color'. \
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
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

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name,opt.classes)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def apply_adjustments(tensor, p=0.8):
    """
    根据概率p调整给定3D Tensor的亮度、对比度、饱和度和色调。
    tensor: 一个形状为 (D, H, W) 的3D Tensor，其中D是深度（切片数）。
    p: 应用调整的概率。
    """
    # 检查是否应用变换
    if random.random() < p:
        # 选择调整因子
        brightness_factor = random.uniform(1 - 0.2, 1 + 0.2)  # 亮度调整系数
        contrast_factor = random.uniform(1 - 0.2, 1 + 0.2)  # 对比度调整系数
        saturation_factor = random.uniform(1 - 0.2, 1 + 0.2)  # 饱和度调整系数
        hue_factor = random.uniform(-0.1, 0.1)  # 色调调整系数

        # 假设tensor是 (D, H, W)，并且D代表不同的切片
        # 需要将tensor调整为 (D, C, H, W) 来模拟批处理，其中C=1因为是单通道
        tensor = tensor.unsqueeze(1)  # 增加一个通道维度

        # 应用亮度、对比度、饱和度和色调调整
        adjusted_tensor = TF.adjust_brightness(tensor, brightness_factor)
        adjusted_tensor = TF.adjust_contrast(adjusted_tensor, contrast_factor)
        adjusted_tensor = TF.adjust_saturation(adjusted_tensor, saturation_factor)
        adjusted_tensor = TF.adjust_hue(adjusted_tensor, hue_factor)

        # 移除通道维度，恢复为 (D, H, W)
        adjusted_tensor = adjusted_tensor.squeeze(1)

        return adjusted_tensor
    else:
        return tensor

class GAMMA_sub1_dataset(Dataset):
    def __init__(self,
                 img_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.mode = mode.lower()

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]
        fundus_img_path = os.path.join(self.dataset_root, real_index,real_index + ".jpg")
        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
       # print("fund_shape",fundus_img.shape)
        if fundus_img.shape[0] == 2000:
            #pdb.set_trace()
            #print("前",fundus_img.shape)
            fundus_img = fundus_img[1000 - 967:1000 + 967, 1496 - 978:1496 + 978, :]
            #print("hou",fundus_img.shape)
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomApply([
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
            # ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            #transforms.CenterCrop(400),
            transforms.Resize(384)
        ])

        fundus_img = fundus_img.copy()
        if self.img_transforms is not None:
            fundus_img1 = self.img_transforms(fundus_img)  # 只是处理一下图像
            fundus_img2 = self.img_transforms(fundus_img)
        fundus_img = transform(fundus_img)
        fundus_img3 = [(fundus_img2 / 255.), (fundus_img1 / 255.)]

        if self.mode == 'test':
            return fundus_img, real_index
        if self.mode == "train":
            label = label.argmax()
            return fundus_img3, label

    def __len__(self):
        return len(self.file_list)
class GAMMA_sub1_dataset_vessel(Dataset):
    def __init__(self,
                 img_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.mode = mode.lower()

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}
            #pdb.set_trace()
            self.file_list = [[f[:-4], label[int(f[:-4])]] for f in os.listdir(dataset_root)]
            #pdb.set_trace()
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        # if filelists is not None:
        #     pdb.set_trace()
        #     self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        #pdb.set_trace()
        real_index, label = self.file_list[idx]
        vessel_img_path = os.path.join(self.dataset_root,real_index + ".jpg")

        #vessel_img = Image.open(vessel_img_path).convert('L')
        vessel_img = cv2.imread(vessel_img_path, cv2.IMREAD_GRAYSCALE)
        #vessel_img = cv2.imread(vessel_img_path)[:, :, ::-1]  # BGR -> RGB
       # print("fund_shape",fundus_img.shape)
        if vessel_img.shape[0] == 2000:
            #pdb.set_trace()
            #print("前",fundus_img.shape)
            vessel_img = vessel_img[1000 - 967:1000 + 967, 1496 - 978:1496 + 978]
            #print("hou",fundus_img.shape)
        transform = transforms.Compose([
            transforms.ToTensor(),

            transforms.Resize(512)
        ])

        vessel_img = vessel_img.copy()
        if self.img_transforms is not None:
            vessel_img1 = self.img_transforms(vessel_img)  # 只是处理一下图像
            vessel_img2 = self.img_transforms(vessel_img)
        vessel_img = transform(vessel_img)
        vessel_img3 = [(vessel_img2 / 255.), (vessel_img1 / 255.)]

        if self.mode == 'test':
            return vessel_img, real_index
        if self.mode == "train":
            label = label.argmax()
            return vessel_img3, label

    def __len__(self):
        #pdb.set_trace()
        return len(self.file_list)
class GAMMA_sub1_dataset_oct(Dataset):
    def __init__(self,
                 img_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.mode = mode.lower()
        # self.oct_transforms = trans.Compose([
        #     trans.ToTensor(),
        #     trans.CenterCrop(384),
        #     trans.RandomHorizontalFlip(),
        #     trans.RandomVerticalFlip()
        # ])
        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]
        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))
        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W
        oct_img = oct_img.transpose(1, 2, 0)  # H, W , 256
        #print("oct_img",oct_img.shape)


        transform = transforms.Compose([

            # transforms.RandomApply([
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
            # ], p=0.8),
            transforms.ToTensor(),
            transforms.CenterCrop(384),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

        oct_img = oct_img.copy()
        #print("oct.shapexxxxxxxxxxxxxx",oct_img.shape)  #(992, 512, 256)
        #print("qian",oct_img.max())
        if self.img_transforms is not None:
            oct_img1 = self.img_transforms(oct_img)  # 只是处理一下图像
            oct_img2 = self.img_transforms(oct_img)
        #print("hou",len(oct_img1))
        oct_img = transform(oct_img)
        #print("hou", len(oct_img1))
        #print(oct_img1.shape)
        oct_img3 = [(oct_img2 / 255.), (oct_img1 / 255.)]
        #oct_img3 = [(oct_img2/255 ), (oct_img1 )]

        if self.mode == 'test':
            return oct_img, real_index
        if self.mode == "train":
            label = label.argmax()
            return oct_img3, label

    def __len__(self):
        return len(self.file_list)
def set_loader(opt):
    # construct data loaders
    mean = (0.3163843, 0.86174834, 0.3641431)
    std = (0.24608557, 0.11123227, 0.26710403)
    trainset_root = "./datasets/gamma/Glaucoma_grading/training/multi-modality_images"
    filelists = os.listdir(trainset_root)
    train_filelists = [[], [], [], [], []]
    val_filelists = [[], [], [], [], []]
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    y = kf.split(filelists)
    count = 0
    for tidx, vidx in y:
        train_filelists[count], val_filelists[count] = np.array(filelists)[tidx], np.array(filelists)[vidx]
        count = count + 1
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.CenterCrop(1400),
        transforms.Resize(1024),
        transforms.RandomHorizontalFlip(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.CenterCrop(400),
        transforms.Resize(384),
        normalize,
    ])

    train_dataset = GAMMA_sub1_dataset(dataset_root=trainset_root,
                                       img_transforms=train_transform,
                                       filelists=np.array(filelists),
                                       label_file='/root/autodl-tmp/SupContrast/datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx')

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader
def set_loader_vessel(opt):
    # construct data loaders
    mean = [0.0371]
    std = [0.1080]
    trainset_root = "./datasets/Vessel/training"
    filelists = os.listdir(trainset_root)
    train_filelists = [[], [], [], [], []]
    val_filelists = [[], [], [], [], []]
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    y = kf.split(filelists)
    count = 0
    for tidx, vidx in y:
        train_filelists[count], val_filelists[count] = np.array(filelists)[tidx], np.array(filelists)[vidx]
        count = count + 1
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        ], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        transforms.CenterCrop(1400),
        transforms.Resize(512),
        transforms.RandomHorizontalFlip(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.CenterCrop(400),
        transforms.Resize(512),
        normalize,
    ])

    train_dataset = GAMMA_sub1_dataset_vessel(dataset_root=trainset_root,
                                       img_transforms=train_transform,
                                       filelists=np.array(filelists),
                                       label_file='/root/autodl-tmp/SupContrast/datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx')

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader
def set_loader_oct(opt):
    # construct data loaders
    # 读取Excel文件
    df = pd.read_excel('means_stds.xlsx')

    # 将'Mean'和'Std'列转换为列表
    means_list = df['Mean'].tolist()
    stds_list = df['Std'].tolist()

    # 将列表转换为元组
    mean = tuple(means_list)
    std = tuple(stds_list)
    mean=[0.2811]
    std=[0.0741]
    trainset_root = "./datasets/gamma/Glaucoma_grading/training/multi-modality_images"
    filelists = os.listdir(trainset_root)
    train_filelists = [[], [], [], [], []]
    val_filelists = [[], [], [], [], []]
    # kf = KFold(n_splits=5, shuffle=True, random_state=10)
    # y = kf.split(filelists)
    # count = 0
    # for tidx, vidx in y:
    #     train_filelists[count], val_filelists[count] = np.array(filelists)[tidx], np.array(filelists)[vidx]
    #     count = count + 1
    #normalize = transforms.Normalize(mean=mean, std=std)


    # train_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
    #     ], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     #transforms.CenterCrop(400),
    #     transforms.Resize(384),
    #     transforms.RandomHorizontalFlip(),
    #     normalize,
    # ])
    normalize = transforms.Normalize(mean=[0.2811], std=[0.0741])
    train_transform = TransformAllSlices3(transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        ], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        transforms.Resize([512,512]),
        transforms.RandomHorizontalFlip(),
        normalize,
    ]))

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(400),
        transforms.Resize(384),
        normalize,
    ])

    train_dataset = GAMMA_sub1_dataset_oct(dataset_root=trainset_root,
                                       img_transforms=train_transform,
                                       filelists=np.array(filelists),
                                       label_file='/root/autodl-tmp/SupContrast/datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx')
    # test_dataset = GAMMA_sub1_dataset_oct(dataset_root='./datasets/gamma/Glaucoma_grading/testing/multi-modality_images',
    #                                 # label_file='./datasets/gamma/Glaucoma_grading/ting/glaucoma_grading_training_GT.xlsx',
    #                                  filelists=None,
    #                                  img_transforms=val_transform,
    #                                 model=test
    #                                  )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader

class myModel_oct(nn.Module):
    def __init__(self, opt):
        super(myModel_oct, self).__init__()
        #self.encoder = timm.create_model('resnet50', pretrained=True, num_classes=0)
        num_input_channels = 128
        self.encoder = timm.create_model('resnet50', pretrained=True, num_classes=3)
        # self.encoder.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.encoder = resnet50_1()
        self.encoder.fc = nn.Identity()  # 将最后的fc层替换为Identity，仅传递特征
        # self.encoder = resnet50_128()
        # self.encoder.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
        )

    def forward(self, data):
        feature = self.encoder(data)
        #feature = torch.flatten(feature, 1)
        feature = torch.nn.functional.normalize(self.head(feature), dim=1)
        return feature

class myModel_oct3(nn.Module):
    def __init__(self, opt):
        super(myModel_oct3, self).__init__()
        #self.encoder = timm.create_model('resnet50', pretrained=True, num_classes=0)
        #num_input_channels = 3
        self.encoder = timm.create_model('resnet50', pretrained=True, num_classes=3)
        #self.encoder.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.encoder = resnet50_1()
        self.encoder.fc = nn.Identity()  # 将最后的fc层替换为Identity，仅传递特征
       # self.encoder = resnet50_128()
      #  self.encoder.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
        )

    def forward(self, data):
        feature = self.encoder(data)
        #feature = torch.flatten(feature, 1)
        feature = torch.nn.functional.normalize(self.head(feature), dim=1)
        return feature
class myModel1(nn.Module):
    def __init__(self, opt):
        super(myModel1, self).__init__()
        self.encoder = timm.create_model('resnet50', pretrained=True, num_classes=3)

        self.encoder.fc = nn.Identity()  # 将最后的fc层替换为Identity，仅传递特征
        #self.encoder = resnet50()
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
        )

    def forward(self, data):
        feature = self.encoder(data)
        # feature = torch.flatten(feature, 1)
        feature = torch.nn.functional.normalize(self.head(feature), dim=1)

        return feature
class myModel_vessel(nn.Module):
    def __init__(self, opt):
        super(myModel_vessel, self).__init__()
        num_input_channels=1
        self.encoder = timm.create_model('resnet50', pretrained=True, num_classes=3)
        self.encoder.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.encoder = resnet50_1()
        self.encoder.fc = nn.Identity()  # 将最后的fc层替换为Identity，仅传递特征
        #self.encoder.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
        )

    def forward(self, data):
        feature = self.encoder(data)
        # feature = torch.flatten(feature, 1)
        feature = torch.nn.functional.normalize(self.head(feature), dim=1)

        return feature

class fusion(nn.Module):
    def __init__(self, opt):
        super(fusion, self).__init__()
        self.OCTbranch = myModel_oct(opt)
        self.fundusbranch = myModel1(opt)
        self.vesselbranch = myModel1(opt)
        
    def forward(self, fundus, OCT, vessel):
        fundus_feature = self.fundusbranch(fundus)
        OCT_feature = self.OCTbranch(OCT)
        vessel_feature = self.vesselbranch(vessel)
        
        return fundus_feature, OCT_feature, vessel_feature


def set_model(opt):
    if opt.classes=="fundus":
        model = myModel1(opt)
    if opt.classes=="oct":
        model=myModel_oct3(oct)
    # if opt.classes=="vessel":
    #     model=myModel_vessel3(oct)
    # model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    if opt.method == 'NTXent':
        criterion = NTXentLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        #pdb.set_trace()
        images = torch.cat([images[0], images[1]], dim=0)# image0 12 3 384 384
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        images = images.contiguous()
        #pdb.set_trace()
        # compute loss  #images [24,3,384,384] 里面有1个anchor 1 个经过旋转变换的，其他的是别的
        #images [24,96,96,256]
        features = model(images)  #feature .shape =[24,128] 24个图片的各自128维的feature
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)  # f1 [12,128] f2 [12,128] 相当于把anchor和经过变幻的图像的feature得到
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  #features [12,2,128]
        if opt.method == 'SupCon':
            pdb.set_trace()
            loss = criterion(features, labels) #labels 12
        elif opt.method == 'SimCLR':
            loss = criterion(features)

        elif opt.method == 'NTXent':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    if opt.classes=="fundus":
        train_loader = set_loader(opt)
    if opt.classes=="oct":
        train_loader = set_loader_oct(opt)
    if opt.classes=="vessel":
        train_loader = set_loader_vessel(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder,'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()