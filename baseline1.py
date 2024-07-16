from __future__ import print_function
import pandas as pd
import os
import pdb
import sys
import argparse
import time
import math
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import torch.nn as nn
from utilbaseline import TwoCropTransform, AverageMeter
from utilbaseline import adjust_learning_rate, warmup_learning_rate
from utilbaseline import set_optimizer, save_model
from networks.baseline_test1 import ResNetBaseline1_,ResNetBaseline2_,MLPBaseline_
from losses import SupConLoss
import torch.optim as optim
from main_ce import TransformAllSlices, TransformAllSlices2
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from MyDatasets import GAMMA_dataset , GAMMA_dataset_oct , GAMMA_dataset_fund, GAMMA_dataset_all,GAMMA_dataset_baseline
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='5,10,15',
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
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--classes', type=str, default='fundus_data',
                        choices=['oct_data', 'fundus_data',"all_data"], help='what class data will be need')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.05,
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

    opt = parser.parse_args()
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

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
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

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    mean = (0.0, 0.0, 0.0)  # 设置一个默认值
    std = (1.0, 1.0, 1.0)    # 设置一个默认值
    mean_fundus = (0.3163843, 0.86174834, 0.3641431)
    std_fundus = (0.24608557, 0.11123227, 0.26710403)
    mean_oct = [0.2811]
    std_oct = [0.0741]
    normalize_fundus = transforms.Normalize(mean=mean_fundus, std=std_fundus)
    normalize_oct = transforms.Normalize(mean=mean_oct, std=std_oct)
    
    
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        if opt.classes == 'fundus_data':
            #mean = (0.04972722, 0.17129958, 0.3469699)
            #std = (0.08022357, 0.16009043, 0.2897089)
            mean = (0.3163843, 0.86174834, 0.3641431)
            std = (0.24608557, 0.11123227, 0.26710403)
        if opt.classes=='oct_data':
            # 读取Excel文件
            df = pd.read_excel('means_stds.xlsx')

            # 将'Mean'和'Std'列转换为列表
            means_list = df['Mean'].tolist()
            stds_list = df['Std'].tolist()

            # 将列表转换为元组
            mean = tuple(means_list)
            std = tuple(stds_list)

            # 打印结果
            # print("means =", means_tuple)
            # print("stds =", stds_tuple)
            # mean = (0.04972722, 0.17129958, 0.3469699)
            # std = (0.08022357, 0.16009043, 0.2897089)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform_fundus = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.CenterCrop(1400),
            transforms.Resize(384),
            transforms.RandomHorizontalFlip(),
            normalize_fundus,
        ])
    
    train_transform_ves = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(1400),
            transforms.Resize(384),
            transforms.RandomHorizontalFlip()
        ])

    
    train_transform_oct = TransformAllSlices(transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
                ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.Resize([360, 360]),
                transforms.RandomHorizontalFlip(),
                normalize_oct,
            ]))


    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        print("error in reading")
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        print("error in reading")
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        # train_dataset = datasets.ImageFolder(root=opt.data_folder,
        #                                     transform=TwoCropTransform(train_transform))
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.CenterCrop(400),
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(400),
            transforms.Resize(256),
            normalize,
        ])
        if opt.classes=='oct_data':
            train_dataset = GAMMA_dataset_oct(dataset_root='./datasets/gamma/Glaucoma_grading/training/multi-modality_images',
                     label_file='./datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx',
                     filelists=None,
                     num_classes=3,
                     mode='train')
            test_dataset = GAMMA_dataset_oct(dataset_root='./datasets/gamma/Glaucoma_grading/testing/multi-modality_images',
                                          label_file='./datasets/gamma/Glaucoma_grading/testing/glaucoma_grading_training_GT.xlsx',
                                          filelists=None,
                                          num_classes=3,
                                          mode='test')
        if opt.classes == 'fundus_data':

            train_dataset = GAMMA_dataset_fund(img_transforms=train_transform,
                dataset_root='./datasets/gamma/Glaucoma_grading/training/multi-modality_images',
                label_file='./datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx',
                filelists=None,
                num_classes=3,
                mode='train')
            test_dataset = GAMMA_dataset_fund(img_transforms=val_transform,
                dataset_root='./datasets/gamma/Glaucoma_grading/testing/multi-modality_images',
                                         label_file='./datasets/gamma/Glaucoma_grading/testing/glaucoma_grading_training_GT.xlsx',
                                         filelists=None,
                                         num_classes=3,
                                         mode='test')
        if opt.classes == 'all_data':
            train_dataset = GAMMA_dataset_baseline(
            dataset_root='./datasets/gamma/Glaucoma_grading/training/multi-modality_images',
            label_file='./datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx',
            funds_transforms= train_transform_fundus,
            vessel_transforms= train_transform_ves,
            img_transforms= train_transform_oct,
            vessel_root='./datasets/Vessel/training',
            filelists=None,
            num_classes=3,
            mode='train')
            test_dataset = GAMMA_dataset_baseline(
            dataset_root='./datasets/gamma/Glaucoma_grading/training/multi-modality_images',
            label_file='./datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx',
            funds_transforms= train_transform_fundus,
            vessel_transforms= train_transform_ves,
            img_transforms= train_transform_oct,
            vessel_root='./datasets/Vessel/training',
            filelists=None,
            num_classes=3,
            mode='test')

    else:
        raise ValueError(opt.dataset)
    #pdb.set_trace()
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
    #     num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader=train_loader
    return train_loader, val_loader

def set_model(opt):
    #原本的model应该是其他任务，我们显示需要做baseline结果的分类任务，就在model里面进行变化，我们首先先使用Dual-Res
    
    #model = SupConResNet(name=opt.model)
    #criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    #if opt.syncBN:
        #model = apex.parallel.convert_syncbn_model(model)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        #if torch.cuda.device_count() > 1:
        #    model.encoder = torch.nn.DataParallel(model.encoder)
        #model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return criterion


def train(train_loader, modelA,modelB,modelC, criterion, optimizer, epoch, opt):
    """one epoch training"""
    modelA.train()
    modelB.train()
    modelC.train()
    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        #pdb.set_trace()   #[368, 3, 32, 32] cifar 10的输入  batch， channel w h
                        #[100, 3, 256, 256] funds 的输入
        #pdb.set_trace()
        #print(type(data))  # 打印data的类型
        #print(len(data))   # 打印data包含的元素数量
        oct_img, fundus_img, labels = data[0][0], data[1][0], data[2]
        #print(type(data[0][0]))
        #print(type(data[1][0]))
        #print(type(data[2]))

        if torch.cuda.is_available():
            images1 = oct_img.cuda(non_blocking=True)
            images2 = fundus_img.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        features1=modelA(images1)
        features2=modelB(images2)
        features = torch.cat([features1, features2], dim=1)  
        #pdb.set_trace()
        category=modelC(features)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # 计算损失
        loss = criterion(category,labels)
        

        # update metric
        losses.update(loss.item(), bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

def main():
    opt = parse_option()
    #先读取好相关的内容，作为参数存储到opt
    
    
    # build data loader
    train_loader,val_loader = set_loader(opt)
    #设置好dataloader，在all的时候读取两种数据，第一维度是一百张的灰度图，第二维度应该是眼球的RGB照片
    
    # build model and criterion
    criterion = set_model(opt)
    
    ResNetBaseline1 = ResNetBaseline1_(name='resnet2', feat_dim=128).cuda()
    ResNetBaseline2 = ResNetBaseline2_(name='resnet50', feat_dim=128).cuda()
    MLPbaseline = MLPBaseline_(name='resnet50', feat_dim=128).cuda()
    
    # 将三个模型的参数合并为一个列表
    all_parameters = list(ResNetBaseline1.parameters()) + list(ResNetBaseline2.parameters()) + list(MLPbaseline.parameters())

    # 创建优化器
    optimizer = optim.Adam(all_parameters, lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    print("Start Training...")
    
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, ResNetBaseline1,ResNetBaseline2,MLPbaseline,criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)


    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(ResNetBaseline1, optimizer, opt, opt.epochs, save_file)
    save_model(ResNetBaseline2, optimizer, opt, opt.epochs, save_file)
    save_model(MLPbaseline, optimizer, opt, opt.epochs, save_file)
    #评估模型的性能
    
    # 评估模型的性能
    ResNetBaseline1.eval()
    ResNetBaseline2.eval()
    MLPbaseline.eval()

    all_predictions = []
    all_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for idx, data in enumerate(train_loader):
            oct_img, fundus_img, labels = data[0][0].to(device), data[1][0].to(device), data[2].to(device)
            features1 = ResNetBaseline1(oct_img)
            features2 = ResNetBaseline2(fundus_img)
            features = torch.cat([features1, features2], dim=1)  
            # pdb.set_trace()
            category = MLPbaseline(features)
            # 获取模型预测的类别
            _, predictions = torch.max(category, 1)

            # 将预测结果和实际标签存储起来
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # 计算Kappa系数
    kappa = cohen_kappa_score(all_labels, all_predictions)
    print(f'Kappa: {kappa:.4f}')

    # 计算F1分数
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'F1 Score: {f1:.4f}')


if __name__ == '__main__':
    main()
