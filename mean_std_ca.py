import pdb
import torch
import numpy as np
import cv2
import os
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import save_image
from torch.utils import data
import torchvision.transforms as trans
import pandas as pd
import torch

class GAMMA_dataset(data.Dataset):
    """
    getitem() output:

    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = trans.Compose([
            trans.ToTensor(),
            trans.RandomResizedCrop(
                256, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip(),
            trans.RandomRotation(30)
        ])  ##fudons image size -->256
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
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))

        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
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
        if self.oct_transforms is not None:
            oct_img = self.oct_transforms(oct_img.copy())

        save_image(fundus_img, 'fundus.png')
        save_image(oct_img[:, :, 0], 'oct.png')

        if self.mode == 'test':
            return {'fundus': fundus_img, 'oct': oct_img, 'label': real_index.astype('int64')}
        if self.mode == "train":
            label = label.argmax()
            return {'fundus': fundus_img, 'oct': oct_img, 'label': label.astype('int64')}

    def __len__(self):
        return len(self.file_list)
dataset=train_dataset = GAMMA_dataset(
                dataset_root='./datasets/gamma/Glaucoma_grading/training/multi-modality_images',
                label_file='./datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx',
                filelists=None,
                num_classes=3,
                mode='train')
oct_list=[]
fund_list=[]
means, stdevs = [], []
# for i in range(1,100):
#     oct=dataset.__getitem__(i)['oct']
#     #fund=dataset.__getitem__(i)['fundus']
#     #pdb.set_trace()
#     oct = oct[:, :, :, np.newaxis]
#     oct_list.append(oct)
#
#
#
#     #fund = fund[:, :, :, np.newaxis]
#     #fund_list.append(fund)
#
# octs = np.concatenate(oct_list, axis=3)
# octs = octs.astype(np.float32)
#
# for i in range(255):
#         pixels = funds[i, :, :, :].ravel()  # 拉成一行
#         means.append(np.mean(pixels))
#         stdevs.append(np.std(pixels))
# print("oct_normMean = {}".format(means))
# print("oct_normStd = {}".format(stdevs))
import pandas as pd

# 读取Excel文件
df = pd.read_excel('means_stds.xlsx')

# 将'Mean'和'Std'列转换为列表
means_list = df['Mean'].tolist()
stds_list = df['Std'].tolist()

# 将列表转换为元组
means_tuple = tuple(means_list)
stds_tuple = tuple(stds_list)

# 打印结果
print("means =", means_tuple)
print("stds =", stds_tuple)

pdb.set_trace()
channels = 255  # 通道数
sums = np.zeros(channels)  # 存储每个通道的累积总和
sq_sums = np.zeros(channels)  # 存储每个通道的平方累积总和
N = 0  # 数据点总数（对于所有通道相同）

for i in range(1, 100):  # 假设有99个数据点
    oct = dataset.__getitem__(i)['oct']
    oct = oct.to(dtype=torch.float32)#.astype(np.float32)  # 确保数据类型为 float32
    if N == 0:  # 如果是第一次迭代，初始化N
        N = oct.shape[1] * oct.shape[2]  # w * h
   # pdb.set_trace()
    # 对每个通道更新累积值
    for c in range(channels):
        sums[c] += oct[c].sum()
        sq_sums[c] += (oct[c].ravel() ** 2).sum()

# 计算每个通道的均值和标准差
means = sums / (N * 99)  # 99为oct的数量
stds = np.sqrt(sq_sums / (N * 99) - means ** 2)

print("Means per channel: ", means)
print("Stds per channel: ", stds)
means_np = means
stds_np = stds

# 创建一个DataFrame
df = pd.DataFrame({'Mean': means_np, 'Std': stds_np})

# 保存DataFrame到Excel文件
df.to_excel('means_stds.xlsx', index=False)
#means, stdevs = [], []
#funds = np.concatenate(fund_list, axis=3)
#funds = funds.astype(np.float32)

# for i in range(3):
#     pixels = funds[i, :, :, :].ravel()  # 拉成一行
#     means.append(np.mean(pixels))
#     stdevs.append(np.std(pixels))
# means.reverse()
# stdevs.reverse()
#
# print("fund_normMean = {}".format(means))
# print("fund_normStd = {}".format(stdevs))


# img_h, img_w = 32, 32
#
# img_h, img_w = 32, 48  # 根据自己数据集适当调整，影响不大
# means, stdevs = [], []
# img_list = []
#
# imgs_path = 'D:/database/VOCdevkit/VOC2012/JPEGImages/'
# imgs_path_list = os.listdir(imgs_path)
#
# len_ = len(imgs_path_list)
# i = 0
# for item in imgs_path_list:
#     img = cv2.imread(os.path.join(imgs_path, item))
#     img = cv2.resize(img, (img_w, img_h))
#     img = img[:, :, :, np.newaxis]
#     img_list.append(img)
#     i += 1
#     print(i, '/', len_)
#
# imgs = np.concatenate(img_list, axis=3)
# imgs = imgs.astype(np.float32) / 255.
#
# for i in range(3):
#     pixels = imgs[:, :, i, :].ravel()  # 拉成一行
#     means.append(np.mean(pixels))
#     stdevs.append(np.std(pixels))
#
# # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
# means.reverse()
# stdevs.reverse()
#
# print("normMean = {}".format(means))
# print("normStd = {}".format(stdevs))