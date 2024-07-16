from __future__ import print_function

import os
import sys
import argparse
import time
import math
import random
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupCEResNet
from MyDatasets import GAMMA_dataset , GAMMA_dataset_oct , GAMMA_dataset_fund, GAMMA_dataset_all,GAMMA_dataset_two
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

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
        x = x[::2, :, :]  #现在形状是(128, 512, 512)
        #x = x[125:128, :, :]
        #print(x.shape)
        #x = x.unsqueeze(1)

       # x_resized = F.interpolate(x, size=(360, 360), mode='bilinear', align_corners=False)
       #  x_resized = x_resized.squeeze(1)  # 移除“通道”维度，形状变为(256, 360, 360)
       #  x = x_resized.permute(1, 2, 0)  # 重新排列维度以匹配目标形状

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
class TransformAllSlices2:
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

        #resize
        x = x.permute(2, 0, 1)  # 现在形状是(256, 992, 512)
        x = x[:, 150:662, :]  # 现在形状是(256, 512, 512)
        x = x[::2, :, :]
        #print(x.shape)
        #x = x.unsqueeze(1)

        # x_resized = F.interpolate(x, size=(360, 360), mode='bilinear', align_corners=False)
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

        #resize
        x = x.permute(2, 0, 1)  # 现在形状是(256, 992, 512)
        #x = x[:, 150:662, :]  # 现在形状是(256, 512, 512)
        #x = x[::2, :, :]
        x = x[125:128, :, :]
        #print(x.shape)
        #x = x.unsqueeze(1)

        # x_resized = F.interpolate(x, size=(360, 360), mode='bilinear', align_corners=False)
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
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

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

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        if opt.classes == 'fundus':
            #mean = (0.04972722, 0.17129958, 0.3469699)
            #std = (0.08022357, 0.16009043, 0.2897089)
            mean = (0.3163843, 0.86174834, 0.3641431)
            std = (0.24608557, 0.11123227, 0.26710403)
        if opt.classes=='oct':
            mean = [0.2811]
            std = [0.0741]
        if opt.classes=='two':
            fundus_mean = (0.3163843, 0.86174834, 0.3641431)
            fundus_std = (0.24608557, 0.11123227, 0.26710403)
            oct_mean = [0.2811]
            oct_std = [0.0741]
        if opt.classes=="all":
            mean = (0.3163843, 0.86174834, 0.3641431)
            std = (0.24608557, 0.11123227, 0.26710403)
            fundus_mean = (0.3163843, 0.86174834, 0.3641431)
            fundus_std = (0.24608557, 0.11123227, 0.26710403)
            oct_mean = [0.2811]
            oct_std = [0.0741]
            vessel_mean = [0.0371]
            vessel_std = [0.1080]
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    if opt.classes == "fundus" or opt.classes == "oct" :
        normalize = transforms.Normalize(mean=mean, std=std)

        train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

        val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'path':
        # train_dataset = datasets.ImageFolder(root=opt.data_folder,
        #                                     transform=TwoCropTransform(train_transform))
        if opt.classes =='fundus':
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.CenterCrop(1400),
                transforms.Resize(512),
                transforms.RandomHorizontalFlip(),
                normalize,
            ])

            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(1400),
                transforms.Resize(512),
                normalize,
            ])
        if opt.classes=='oct':
            train_transform = TransformAllSlices(transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
                ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.Resize([512, 512]),
                transforms.RandomHorizontalFlip(),
                normalize,
            ]))

            val_transform = TransformAllSlices2(transforms.Compose([
                transforms.ToTensor(),
                #transforms.CenterCrop(400),
                transforms.Resize(512),
                normalize,
            ]))
            train_dataset = GAMMA_dataset_oct(
                     img_transforms=train_transform,
                     dataset_root='./datasets/gamma/Glaucoma_grading/training/multi-modality_images',
                     label_file='./datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx',
                     filelists=None,
                     num_classes=3,
                     mode='train')
            val_dataset = GAMMA_dataset_oct(img_transforms=val_transform,
                                             dataset_root='./datasets/gamma/Glaucoma_grading/Test/multi-modality_images',
                                             label_file='./datasets/gamma/Glaucoma_grading/Test/glaucoma_grading_testing_GT.xlsx',
                                             filelists=None,
                                             num_classes=3,
                                             mode='test')
        if opt.classes == 'fundus':

            train_dataset = GAMMA_dataset_fund(img_transforms=train_transform,
                dataset_root='./datasets/gamma/Glaucoma_grading/training/multi-modality_images',
                label_file='./datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx',
                filelists=None,
                num_classes=3,
                mode='train')
            val_dataset = GAMMA_dataset_fund(img_transforms=val_transform,
                dataset_root='./datasets/gamma/Glaucoma_grading/Test/multi-modality_images',
                                         label_file='./datasets/gamma/Glaucoma_grading/Test/glaucoma_grading_testing_GT.xlsx',
                                         filelists=None,
                                         num_classes=3,
                                         mode='test')
        if opt.classes =='two':
            oct_normalize = transforms.Normalize(mean=oct_mean, std=oct_std)
            oct_train_transform = TransformAllSlices(transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
                ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.Resize([512, 512]),
                transforms.RandomHorizontalFlip(),
                oct_normalize,
            ]))

            oct_val_transform = TransformAllSlices2(transforms.Compose([
                transforms.ToTensor(),
                # transforms.CenterCrop(400),
                transforms.Resize(512),
                oct_normalize,
            ]))
            fundus_normalize = transforms.Normalize(mean=fundus_mean, std=fundus_std)
            funds_train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.CenterCrop(1400),
                transforms.Resize(512),
                transforms.RandomHorizontalFlip(),
                fundus_normalize,
            ])

            funds_val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(1400),
                transforms.Resize(512),
                fundus_normalize,
            ])


            train_dataset = GAMMA_dataset_two(
                img_transforms=oct_train_transform,
                funds_transforms=funds_train_transform,
               # vessel_transforms=vessel_train_transform,
                dataset_root='./datasets/gamma/Glaucoma_grading/training/multi-modality_images',
               # vessel_root='/root/autodl-tmp/SupContrast/datasets/Vessel/training',
                label_file='./datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx',
                filelists=None,
                num_classes=3,
                mode='train')
            val_dataset = GAMMA_dataset_two(img_transforms=oct_val_transform,
                                            funds_transforms=funds_val_transform,
                                           # vessel_transforms=vessel_val_transform,
                                            dataset_root='./datasets/gamma/Glaucoma_grading/Test/multi-modality_images',
                                            #vessel_root='/root/autodl-tmp/SupContrast/datasets/Vessel/testing',
                                            label_file='./datasets/gamma/Glaucoma_grading/Test/glaucoma_grading_testing_GT.xlsx',
                                            filelists=None,
                                            num_classes=3,
                                            mode='test')
        if opt.classes == 'all':
            oct_normalize = transforms.Normalize(mean=oct_mean, std=oct_std)
            oct_train_transform = TransformAllSlices(transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
                ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.Resize([512, 512]),
                transforms.RandomHorizontalFlip(),
                oct_normalize,
            ]))

            oct_val_transform = TransformAllSlices2(transforms.Compose([
                transforms.ToTensor(),
                # transforms.CenterCrop(400),
                transforms.Resize(512),
                oct_normalize,
            ]))
            fundus_normalize = transforms.Normalize(mean=fundus_mean, std=fundus_std)
            funds_train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.CenterCrop(1400),
                transforms.Resize(512),
                transforms.RandomHorizontalFlip(),
                fundus_normalize,
            ])

            funds_val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(1400),
                transforms.Resize(512),
                fundus_normalize,
            ])

            vessel_normalize = transforms.Normalize(mean=vessel_mean, std=vessel_std)
            vessel_train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
                ], p=0.8),
                #transforms.RandomGrayscale(p=0.2),
                transforms.CenterCrop(1400),
                transforms.Resize(512),
                transforms.RandomHorizontalFlip(),
                vessel_normalize,
            ])

            vessel_val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(1400),
                transforms.Resize(512),
                vessel_normalize,
            ])
            train_dataset = GAMMA_dataset_all(
                img_transforms=oct_train_transform,
                funds_transforms=funds_train_transform,
                vessel_transforms=vessel_train_transform,
                dataset_root='./datasets/gamma/Glaucoma_grading/training/multi-modality_images',
                vessel_root='/root/autodl-tmp/SupContrast/datasets/Vessel/training',
                label_file='./datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx',
                filelists=None,
                num_classes=3,
                mode='train')
            val_dataset = GAMMA_dataset_all(img_transforms=oct_val_transform,
                                            funds_transforms=funds_val_transform,
                                            vessel_transforms=vessel_val_transform,
                                             dataset_root='./datasets/gamma/Glaucoma_grading/Test/multi-modality_images',
                                            vessel_root='/root/autodl-tmp/SupContrast/datasets/Vessel/testing',
                                             label_file='./datasets/gamma/Glaucoma_grading/Test/glaucoma_grading_testing_GT.xlsx',
                                             filelists=None,
                                             num_classes=3,
                                             mode='test')
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=14, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
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
    top1 = AverageMeter()

    end = time.time()
    if opt.classes == "all":
        for idx, (oct_images,fundus_images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)

            oct_images = oct_images.cuda(non_blocking=True)
            fundus_images = fundus_images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

            # compute loss
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))
                sys.stdout.flush()
    else:
        for idx, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

            # compute loss
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1))
                sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc = validate(val_loader, model, criterion, opt)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
