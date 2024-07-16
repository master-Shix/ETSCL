# Please Rewrite Dataset Class here
import os
import pdb

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import save_image
from torch.utils import data
import torchvision.transforms as trans
from torchvision import transforms, datasets
class GAMMA_dataset_oct(data.Dataset):
    """
    getitem() output:

    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 img_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        # self.img_transforms = trans.Compose([
        #     trans.ToTensor(),
        #     trans.RandomResizedCrop(
        #         256, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
        #     trans.RandomHorizontalFlip(),
        #     trans.RandomVerticalFlip(),
        #     trans.RandomRotation(30)
        # ])  ##fudons image size -->256
        self.oct_transforms2=img_transforms
        self.oct_transforms = trans.Compose([
            trans.ToTensor(),
            trans.CenterCrop(512),
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip()
        ])
        self.mode = mode.lower()
        self.num_classes = num_classes

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        #fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))

        #fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W
        oct_img = oct_img.transpose(1, 2, 0)  # H, W , 255

        #if self.img_transforms is not None:
            #fundus_img = self.img_transforms(fundus_img.copy())
        if self.oct_transforms2 is not None:
            oct_img1 = self.oct_transforms2(oct_img.copy())
            oct_img2 = self.oct_transforms2(oct_img.copy())

        oct_img3 = [(oct_img2 / 255.), (oct_img1 / 255.)]

        #save_image(fundus_img, 'fundus.png')
        #save_image(oct_img[:, :, 0], 'oct.png')
        if self.mode == 'test':
            #return {'fundus': fundus_img, 'label': real_index.astype('int64')}
            oct_img = self.oct_transforms2(oct_img)
            # print(fundus_img)
            oct_img4 = oct_img / 255.0
            label = label.argmax()
            # return {'fundus': fundus_img, 'label': real_index.astype('int64')}
            return oct_img4, label.astype('int64')
            #return fundus_img, label.astype('int64')
        if self.mode == "train":
            label = label.argmax()
            return oct_img3,  label.astype('int64')
        # if self.mode == 'test':
        #     #return {'fundus': fundus_img, 'oct': oct_img, 'label': real_index.astype('int64')}
        #     return { 'oct': oct_img, 'label': real_index.astype('int64')}
        # if self.mode == "train":
        #     label = label.argmax()
        #     return {'oct': oct_img, 'label': label.astype('int64')}

    def __len__(self):
        return len(self.file_list)
class GAMMA_dataset_fund(data.Dataset):
    """
    getitem() output:

    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 img_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms2=img_transforms
        self.img_transforms = trans.Compose([
            trans.ToTensor(),
            trans.RandomResizedCrop(
                256, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
            # trans.RandomHorizontalFlip(),
            # trans.RandomVerticalFlip(),
            # trans.RandomRotation(30)
            transforms.RandomGrayscale(p=0.2),
            transforms.CenterCrop(400),
            transforms.Resize(384)
        ])  ##fudons image size -->256
        # self.oct_transforms = trans.Compose([
        #     trans.ToTensor(),
        #     trans.CenterCrop(512),
        #     trans.RandomHorizontalFlip(),
        #     trans.RandomVerticalFlip()
        # ])
        self.mode = mode.lower()
        self.num_classes = num_classes

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}
           # for f in os.listdir(dataset_root):
           #     print("label的东西",f,label[int(f)],label[int(f)].argmax())
            print(dataset_root)
            #pdb.set_trace()
            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            #self.file_list = [[f, None] for f in os.listdir(dataset_root)]
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}
            # for f in os.listdir(dataset_root):
            #     print(f,label[int(f)])
            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        # oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
        #                          key=lambda x: int(x.strip("_")[0]))

        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        # oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
        #                           cv2.IMREAD_GRAYSCALE)
        # oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")

        # for k, p in enumerate(oct_series_list):
        #     oct_img[k] = cv2.imread(
        #         os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        # oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W
        # oct_img = oct_img.transpose(1, 2, 0)  # H, W , 255
        if fundus_img.shape[0] == 2000:
            #pdb.set_trace()
            #print("前",fundus_img.shape)
            fundus_img = fundus_img[1000 - 967:1000 + 967, 1496 - 978:1496 + 978, :]
        fundus_img = fundus_img.copy()
        if self.img_transforms2 is not None:
            fundus_img1 = self.img_transforms2(fundus_img)  # 只是处理一下图像
            fundus_img2 = self.img_transforms2(fundus_img)

        fundus_img3 = [(fundus_img2 / 255.), (fundus_img1 / 255.)]

        # if self.img_transforms is not None:
        #     fundus_img = self.img_transforms(fundus_img.copy())

        # if self.oct_transforms is not None:
        #     oct_img = self.oct_transforms(oct_img.copy())
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomApply([
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
            # ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.CenterCrop(1400),
            transforms.Resize(384)
        ])
        #save_image(fundus_img, 'fundus.png')
        #save_image(oct_img[:, :, 0], 'oct.png')

        if self.mode == 'test':
            fundus_img = self.img_transforms2(fundus_img)
            #print(fundus_img)
            fundus_img4=fundus_img/255.0
            label = label.argmax()
            #return {'fundus': fundus_img, 'label': real_index.astype('int64')}
            return fundus_img4, label.astype('int64')
        if self.mode == "train":
            #fundus_img = fundus_img.copy()


            label = label.argmax()
            return fundus_img3,  label.astype('int64')
           # return {'fundus': fundus_img, 'label': label.astype('int64')}

    def __len__(self):
        return len(self.file_list)
    
    
class GAMMA_dataset(data.Dataset):
    """
    getitem() output:

    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 dataset_root,
                 vessel_dataset_root,
                 img_transforms,
                 ves_transforms,
                 oct_transforms,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.vessel_dataset_root = vessel_dataset_root
        self.img_transforms = img_transforms
        self.ves_transforms = ves_transforms
        self.oct_transforms = oct_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes

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

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))
        vessel_img_path = os.path.join(self.vessel_dataset_root, real_index+".jpg")

        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        vessel_img = cv2.imread(vessel_img_path)[:,:,::-1] #for test only
        #vessel_img = cv2.imread(vessel_img_path, cv2.IMREAD_GRAYSCALE)
        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W
        oct_img = oct_img.transpose(1, 2, 0)  # H, W , 255

        if self.img_transforms is not None:
            fundus_img = self.img_transforms(fundus_img.copy())
        if self.ves_transforms is not None:
            vessel_img = self.ves_transforms(vessel_img.copy())
        if self.oct_transforms is not None:
            oct_img = self.oct_transforms(oct_img.copy())

        save_image(fundus_img, 'fundus.png')
        save_image(vessel_img, 'vessel.png')
        
        if self.mode == 'test':
            return fundus_img,  oct_img, vessel_img, label.astype('int64')
        if self.mode == "train":
            label = label.argmax()
            return fundus_img, oct_img, vessel_img, label.astype('int64')


    def __len__(self):
        return len(self.file_list)

class GAMMA_dataset_all(data.Dataset):
    """
        getitem() output:

        	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

            oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
        """

    def __init__(self,
                 img_transforms,
                 funds_transforms,
                 vessel_transforms,
                 dataset_root,
                 vessel_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.vessel_root=vessel_root
        self.oct_transforms2 = img_transforms
        self.img_transforms2 = funds_transforms
        self.vessel_transforms=vessel_transforms
        self.oct_transforms = trans.Compose([
            trans.ToTensor(),
            trans.CenterCrop(512),
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip()
        ])
        self.mode = mode.lower()
        self.num_classes = num_classes

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))

        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB

        if fundus_img.shape[0] == 2000:
            fundus_img = fundus_img[1000 - 967:1000 + 967, 1496 - 978:1496 + 978, :]
        fundus_img = fundus_img.copy()
        if self.img_transforms2 is not None:
            fundus_img1 = self.img_transforms2(fundus_img)  # 只是处理一下图像
            fundus_img2 = self.img_transforms2(fundus_img)

        fundus_img3 = [(fundus_img2 / 255.), (fundus_img1 / 255.)]
        # if self.mode == "test":
        #     vessel_img_path = os.path.join(self.vessel_root, real_index + ".jpg")
        #     real_index=real_index-100
        vessel_img_path = os.path.join(self.vessel_root, real_index + ".jpg")
        #print("vessel.path",vessel_img_path)
        vessel_img = cv2.imread(vessel_img_path, cv2.IMREAD_GRAYSCALE)
        if vessel_img.shape[0] == 2000:
            vessel_img = vessel_img[1000 - 967:1000 + 967, 1496 - 978:1496 + 978]
        vessel_img = vessel_img.copy()
        if self.vessel_transforms is not None:
            vessel_img1 = self.vessel_transforms(vessel_img)  # 只是处理一下图像
            vessel_img2 = self.vessel_transforms(vessel_img)
        vessel_img3 = [(vessel_img2 / 255.), (vessel_img1 / 255.)]


        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W
        oct_img = oct_img.transpose(1, 2, 0)  # H, W , 255

        # if self.img_transforms is not None:
        # fundus_img = self.img_transforms(fundus_img.copy())
        if self.oct_transforms2 is not None:
            oct_img1 = self.oct_transforms2(oct_img.copy())
            oct_img2 = self.oct_transforms2(oct_img.copy())

        oct_img3 = [(oct_img2 / 255.), (oct_img1 / 255.)]


        if self.mode == 'test':
            # return {'fundus': fundus_img, 'label': real_index.astype('int64')}
            oct_img = self.oct_transforms2(oct_img)

            oct_img4 = oct_img / 255.0
            label = label.argmax()

            fundus_img = self.img_transforms2(fundus_img)
            fundus_img4 = fundus_img / 255.0

            transform = transforms.Compose([
                transforms.ToTensor(),

                transforms.Resize(512)
            ])
            vessel_img4 = transform(vessel_img)
            vessel_img4=vessel_img4 /255.0

            # return {'fundus': fundus_img, 'label': real_index.astype('int64')}
            return oct_img4,fundus_img4,  vessel_img4,label.astype('int64')
            # return fundus_img, label.astype('int64')
        if self.mode == "train":
            label = label.argmax()
            return oct_img3,fundus_img3, vessel_img3,label.astype('int64')
        # if self.mode == 'test':
        #     #return {'fundus': fundus_img, 'oct': oct_img, 'label': real_index.astype('int64')}
        #     return { 'oct': oct_img, 'label': real_index.astype('int64')}
        # if self.mode == "train":
        #     label = label.argmax()
        #     return {'oct': oct_img, 'label': label.astype('int64')}

    def __len__(self):
        return len(self.file_list)



class GAMMA_dataset_two(data.Dataset):
    """
        getitem() output:

        	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

            oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
        """

    def __init__(self,
                 img_transforms,
                 funds_transforms,
                 #vessel_transforms,
                 dataset_root,
                 #vessel_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        #self.vessel_root=vessel_root
        self.oct_transforms2 = img_transforms
        self.img_transforms2 = funds_transforms
        #self.vessel_transforms=vessel_transforms
        self.oct_transforms = trans.Compose([
            trans.ToTensor(),
            trans.CenterCrop(512),
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip()
        ])
        self.mode = mode.lower()
        self.num_classes = num_classes

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))

        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB

        if fundus_img.shape[0] == 2000:
            fundus_img = fundus_img[1000 - 967:1000 + 967, 1496 - 978:1496 + 978, :]
        fundus_img = fundus_img.copy()
        if self.img_transforms2 is not None:
            fundus_img1 = self.img_transforms2(fundus_img)  # 只是处理一下图像
            fundus_img2 = self.img_transforms2(fundus_img)

        fundus_img3 = [(fundus_img2 / 255.), (fundus_img1 / 255.)]
        # if self.mode == "test":
        #     vessel_img_path = os.path.join(self.vessel_root, real_index + ".jpg")
        #     real_index=real_index-100
        #vessel_img_path = os.path.join(self.vessel_root, real_index + ".jpg")
        #print("vessel.path",vessel_img_path)
        #vessel_img = cv2.imread(vessel_img_path, cv2.IMREAD_GRAYSCALE)
        #if vessel_img.shape[0] == 2000:
        #    vessel_img = vessel_img[1000 - 967:1000 + 967, 1496 - 978:1496 + 978]
        #vessel_img = vessel_img.copy()
        # if self.vessel_transforms is not None:
        #     vessel_img1 = self.vessel_transforms(vessel_img)  # 只是处理一下图像
        #     vessel_img2 = self.vessel_transforms(vessel_img)
        # vessel_img3 = [(vessel_img2 / 255.), (vessel_img1 / 255.)]


        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W
        oct_img = oct_img.transpose(1, 2, 0)  # H, W , 255

        # if self.img_transforms is not None:
        # fundus_img = self.img_transforms(fundus_img.copy())
        if self.oct_transforms2 is not None:
            oct_img1 = self.oct_transforms2(oct_img.copy())
            oct_img2 = self.oct_transforms2(oct_img.copy())

        oct_img3 = [(oct_img2 / 255.), (oct_img1 / 255.)]


        if self.mode == 'test':
            # return {'fundus': fundus_img, 'label': real_index.astype('int64')}
            oct_img = self.oct_transforms2(oct_img)

            oct_img4 = oct_img / 255.0
            label = label.argmax()

            fundus_img = self.img_transforms2(fundus_img)
            fundus_img4 = fundus_img / 255.0

            transform = transforms.Compose([
                transforms.ToTensor(),

                transforms.Resize(512)
            ])
            #vessel_img4 = transform(vessel_img)
            #vessel_img4=vessel_img4 /255.0

            # return {'fundus': fundus_img, 'label': real_index.astype('int64')}
            return oct_img4,fundus_img4,  label.astype('int64')
            # return fundus_img, label.astype('int64')
        if self.mode == "train":
            label = label.argmax()
            return oct_img3,fundus_img3,label.astype('int64')
        # if self.mode == 'test':
        #     #return {'fundus': fundus_img, 'oct': oct_img, 'label': real_index.astype('int64')}
        #     return { 'oct': oct_img, 'label': real_index.astype('int64')}
        # if self.mode == "train":
        #     label = label.argmax()
        #     return {'oct': oct_img, 'label': label.astype('int64')}

    def __len__(self):
        return len(self.file_list)
class GAMMA_dataset_baseline(data.Dataset):
    """
        getitem() output:

        	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

            oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
        """

    def __init__(self,
                 img_transforms,
                 funds_transforms,
                 vessel_transforms,
                 dataset_root,
                 vessel_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.vessel_root=vessel_root
        self.oct_transforms2 = img_transforms
        self.img_transforms2 = funds_transforms
        self.vessel_transforms=vessel_transforms
        self.oct_transforms = trans.Compose([
            trans.ToTensor(),
            trans.CenterCrop(512),
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip()
        ])
        self.mode = mode.lower()
        self.num_classes = num_classes

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))

        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB

        if fundus_img.shape[0] == 2000:
            fundus_img = fundus_img[1000 - 967:1000 + 967, 1496 - 978:1496 + 978, :]
        fundus_img = fundus_img.copy()
        if self.img_transforms2 is not None:
            fundus_img1 = self.img_transforms2(fundus_img)  # 只是处理一下图像
            fundus_img2 = self.img_transforms2(fundus_img)

        fundus_img3 = [(fundus_img2 / 255.), (fundus_img1 / 255.)]
        # if self.mode == "test":
        #     vessel_img_path = os.path.join(self.vessel_root, real_index + ".jpg")
        #     real_index=real_index-100
        vessel_img_path = os.path.join(self.vessel_root, real_index + ".jpg")
        #print("vessel.path",vessel_img_path)
        vessel_img = cv2.imread(vessel_img_path, cv2.IMREAD_GRAYSCALE)
        if vessel_img.shape[0] == 2000:
            vessel_img = vessel_img[1000 - 967:1000 + 967, 1496 - 978:1496 + 978]
        vessel_img = vessel_img.copy()
        if self.vessel_transforms is not None:
            vessel_img1 = self.vessel_transforms(vessel_img)  # 只是处理一下图像
            vessel_img2 = self.vessel_transforms(vessel_img)
        vessel_img3 = [(vessel_img2 / 255.), (vessel_img1 / 255.)]


        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W
        oct_img = oct_img.transpose(1, 2, 0)  # H, W , 255

        # if self.img_transforms is not None:
        # fundus_img = self.img_transforms(fundus_img.copy())
        if self.oct_transforms2 is not None:
            oct_img1 = self.oct_transforms2(oct_img.copy())
            oct_img2 = self.oct_transforms2(oct_img.copy())

        oct_img3 = [(oct_img2 / 255.), (oct_img1 / 255.)]
        


        if self.mode == 'test':
            # return {'fundus': fundus_img, 'label': real_index.astype('int64')}
            oct_img = self.oct_transforms2(oct_img)

            oct_img4 = oct_img / 255.0
            label = label.argmax()

            fundus_img = self.img_transforms2(fundus_img)
            fundus_img4 = fundus_img / 255.0

            transform = transforms.Compose([
                transforms.ToTensor(),

                transforms.Resize(512)
            ])
            vessel_img4 = transform(vessel_img)
            vessel_img4=vessel_img4 /255.0

            # return {'fundus': fundus_img, 'label': real_index.astype('int64')}
            return oct_img4,fundus_img4,label.astype('int64')
            # return fundus_img, label.astype('int64')
        if self.mode == "train":
            label = label.argmax()
            return oct_img3,fundus_img3,label.astype('int64')
        # if self.mode == 'test':
        #     #return {'fundus': fundus_img, 'oct': oct_img, 'label': real_index.astype('int64')}
        #     return { 'oct': oct_img, 'label': real_index.astype('int64')}
        # if self.mode == "train":
        #     label = label.argmax()
        #     return {'oct': oct_img, 'label': label.astype('int64')}

    def __len__(self):
        return len(self.file_list)
    
    
