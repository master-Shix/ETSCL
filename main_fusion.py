from __future__ import print_function

import sys
import argparse
import time
import math
import torch
import torch.backends.cudnn as cudnn
from networks.resnet_big import myModel_fundus, myModel_oct, myModel_vessel
from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import DSClassifier, MultiLinearClassfier2
from main_ce import TransformAllSlices, TransformAllSlices2
from losses import ce_loss
from torchvision import transforms
from MyDatasets import GAMMA_dataset
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from sklearn.metrics import cohen_kappa_score
import logging

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        ### create a logger
        self.__logger = logging.getLogger(logger_name)

        ### set the log level
        self.__logger.setLevel(log_level)

        ### create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        ### create a handler to print on console
        console_handler = logging.StreamHandler()

        ### define the output format of handlers
        formatter = logging.Formatter('[%(asctime)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        ### add handler to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger

logger = Logger('test', 'myLogger', logging.DEBUG).get_log()

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-1,
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
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100','path'], help='dataset')
    parser.add_argument('--classes', type=str, default='fundus',
                        choices=['fundus', 'all', 'oct'], help='dataset')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt_fundus', type=str, default='./save/SupCon/path_models/SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_12_temp_0.05_trial_0_0922_thick384_color_cosine/fundus/ckpt_epoch_10.pth',
                        help='path to pre-trained fundus model')
    parser.add_argument('--ckpt_OCT', type=str, default='./save/SupCon/path_models/SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_12_temp_0.05_trial_0_0922_thick384_color_cosine/oct/ckpt_epoch_10.pth',
                        help='path to pre-trained OCT model')
    parser.add_argument('--ckpt_vessel', type=str, default='./save/SupCon/path_models/SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_12_temp_0.05_trial_0_0922_thick384_color_cosine/fundus/ckpt_epoch_10.pth',
                        help='path to pre-trained vessel model')
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


def set_loader(opt):
    # construct data loade
    mean_fundus = (0.3163843, 0.86174834, 0.3641431)
    std_fundus = (0.24608557, 0.11123227, 0.26710403)
    mean_oct = [0.2811]
    std_oct = [0.0741]
            
    normalize_fundus = transforms.Normalize(mean=mean_fundus, std=std_fundus)
    normalize_oct = transforms.Normalize(mean=mean_oct, std=std_oct)
    
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

    val_transform_fundus = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(1400),
            transforms.Resize(384),
            normalize_fundus,
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

    val_transform_oct = TransformAllSlices2(transforms.Compose([
                transforms.ToTensor(),
                #transforms.CenterCrop(400),
                transforms.Resize(360),
                normalize_oct,
            ]))

    train_dataset = GAMMA_dataset(
        dataset_root='./datasets/gamma/Glaucoma_grading/training/multi-modality_images',
        vessel_dataset_root='./datasets/Vessel/training',
        label_file='./datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx',
        img_transforms= train_transform_fundus,
        ves_transforms= train_transform_ves,
        oct_transforms= train_transform_oct,
        filelists=None,
        num_classes=3,
        mode='train')
    
    '''
    val_dataset = GAMMA_dataset(img_transforms=val_transform_fundus,
                                oct_transforms=val_transform_oct,
                                ves_transforms=train_transform_ves,
                                vessel_dataset_root='./datasets/Vessel/testing',
                                dataset_root='./datasets/gamma/Glaucoma_grading/Test/multi-modality_images',
                                label_file='./datasets/gamma/Glaucoma_grading/Test/glaucoma_grading_testing_GT.xlsx',
                                filelists=None,
                                num_classes=3,
                                mode='test')
    '''

    train_sampler = None
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    #val_loader = DataLoader(
    #    val_dataset, batch_size=12, shuffle=False,
    #    num_workers=8, pin_memory=True)

    return train_loader


def Trans_state(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict
    

def set_model(opt):
    
    model_fundus = myModel_fundus(opt)
    model_vessel = myModel_vessel(opt)
    model_oct = myModel_oct(opt)
    #criterion = ce_loss
    #classifier = DSClassifier(feat_dim=opt.feat_dim, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = MultiLinearClassfier2()
    
    ckpt_fundus = torch.load(opt.ckpt_fundus, map_location='cpu')
    ckpt_OCT = torch.load(opt.ckpt_OCT, map_location='cpu')
    ckpt_vessel = torch.load(opt.ckpt_vessel, map_location='cpu')
    
    state_dict_fundus = Trans_state(ckpt_fundus['model'])
    state_dict_OCT = Trans_state(ckpt_OCT['model'])
    state_dict_vessel = Trans_state(ckpt_vessel['model'])
    
    model_fundus.load_state_dict(state_dict_fundus)
    model_vessel.load_state_dict(state_dict_vessel)
    model_oct.load_state_dict(state_dict_OCT)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model_fundus = DataParallel(model_fundus, [0,1])
            model_oct = DataParallel(model_oct, [0,1])
            model_vessel = DataParallel(model_vessel, [0,1])
            classifier = DataParallel(classifier, [0,1])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_fundus = model_fundus.to(device)
        model_oct = model_oct.to(device)
        model_vessel = model_vessel.to(device)
        classifier = classifier.to(device)
        #criterion = criterion.cuda()
        cudnn.benchmark = True
        
    else:
        raise NotImplementedError('This code requires GPU')

    return model_fundus, model_oct, model_vessel, classifier, criterion


def train(train_loader, model_fundus, model_oct, model_vessel, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model_fundus.eval()
    model_vessel.eval()
    model_oct.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (fundus, oct, vessel, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        fundus = fundus.cuda(non_blocking=True)
        oct = oct.cuda(non_blocking=True)
        vessel = vessel.cuda(non_blocking= True)
        labels = labels.cuda(non_blocking=True)
        
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            oct_features = model_oct(oct)
            fundus_features = model_fundus(fundus)
            vessel_features = model_vessel(vessel)
        #output_oct, output_fundus, output_vessel, output_combined = classifier(oct_features.detach(), fundus_features.detach(), vessel_features.detach())
        output_combined = classifier(oct_features.detach(), fundus_features.detach())
        loss = criterion(output_combined, labels)
        #loss = criterion(labels, output_oct, opt.n_cls, epoch, opt.annealing_epoch)+\
        
        #       criterion(labels, output_fundus, opt.n_cls, epoch, opt.annealing_epoch)+\
        
        #       criterion(labels, output_vessel, opt.n_cls, epoch, opt.annealing_epoch)+\
        
        #       criterion(labels, output_combined, opt.n_cls, epoch, opt.annealing_epoch) 

        # update metric
        losses.update(loss.item(), bsz)
        #pdb.set_trace()
        acc1 = accuracy(output_combined, labels, topk=(1,))
        top1.update(acc1[0]/100, bsz)

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
    train_loader = set_loader(opt)

    # build model and criterion

    model_fundus, model_oct, model_vessel, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    print('Start Training')
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model_fundus, model_oct, model_vessel, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
    
        print(f'Train epoch: {epoch}, Accuracy:{acc.item()}, Loss:{loss}')
        logger.info(f'Train epoch: {epoch}, Accuracy:{acc.item()}, Loss:{loss}')

        ## eval for one epoch
        #loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        #if val_acc > best_acc:
        #    best_acc = val_acc
        #torch.cuda.empty_cache()

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
