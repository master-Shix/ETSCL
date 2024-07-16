import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from PIL import Image

# 设置使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batchsize = 4
oct_img_size = [512, 512]
image_size = 256
iters = 1000
val_ratio = 0.1

trainset_root = "/root/autodl-tmp/SupContrast/datasets/gamma/Glaucoma_grading/training/multi-modality_images"
test_root = "/root/autodl-tmp/SupContrast/datasets/gamma/Glaucoma_grading/Test/multi-modality_images"
num_workers = 4
init_lr = 1e-4
optimizer_type = "adam"

filelists = os.listdir(trainset_root)
train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=42)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))


class GAMMA_sub1_dataset(Dataset):
    def __init__(self,
                 img_transforms,
                 oct_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.oct_transforms = oct_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes


        if self.mode == 'train':
            label = {str(row['data']).zfill(4): row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}
            
            #print(label)
            #print(os.listdir(dataset_root))
            label_data = pd.read_excel(label_file)
            for _, row in label_data.iterrows():
                print(row['data'], type(row['data']))
                print(row[1:].values, type(row[1:].values))
                break  # 只打印一个示例，你可以注释掉这一行以打印更多示例

            
            self.file_list = [[str(f), label[(f)]] for f in os.listdir(dataset_root) if str(f) in label]

        elif self.mode == "test":
            self.file_list = [[str(f), None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

        print("Final Dataset Length:", len(self.file_list))

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index +".jpg")
        #print("Length of training dataset fundus ooo: ", fundus_img_path)
        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                key=lambda x: int(x.strip("_")[0]))

        #fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        #print("Length of training dataset fundus number: ", len(fundus_img))
        #oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
        #                          cv2.IMREAD_GRAYSCALE)
        #oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")

        #for k, p in enumerate(oct_series_list):
        #    oct_img[k] = cv2.imread(
        #        os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        
        fundus_img = Image.fromarray(cv2.imread(fundus_img_path)[:, :, ::-1])  # BGR -> RGB
        #oct_series_0 = Image.fromarray(cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
        #                                   cv2.IMREAD_GRAYSCALE))
        oct_img = [Image.fromarray(cv2.imread(os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_UNCHANGED).astype(np.uint8)) for p in oct_series_list]
        #oct_img = torch.stack(oct_img)
        # 将fundus_img转换为Tensor对象
        #print("转换前的shape是多少呢？")
        #print(fundus_img.size)
        fundus_img = transforms.ToTensor()(fundus_img)
        #print("转换后的shape是多少呢？")
        #print(fundus_img.shape)
        #print(fundus_img)
        # 将oct_img列表中的每个图像转换为Tensor对象
        #oct_img = [transforms.ToTensor()(img) for img in oct_img]
        # 堆叠oct_img中的所有Tensor对象
        # 在__getitem__中确保 oct_img 是 Tensor 对象
        oct_img = torch.stack([transforms.ToTensor()(img) for img in oct_img], dim=0)


        
        
        if self.img_transforms is not None:
            fundus_img = self.img_transforms(fundus_img)
        if self.oct_transforms is not None:
            oct_img = self.oct_transforms(oct_img)

        #fundus_img = fundus_img.transpose(2, 0, 1)  # H, W, C -> C, H, W
        #fundus_img = fundus_img.transpose(0, 1)
        #print("OCT_img大小")
        #print(oct_img.shape)
        #oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W
        oct_img = oct_img.squeeze(1)  # D, H, W, 1 -> D, H, W，torch.Size([256, 512, 512])
        oct_img = oct_img.permute(0, 2, 1)  # 交换第二和第三维度，torch.Size([256, 512, 512])
        #print("OCT_img裁剪后的大小")
        #print(oct_img.shape)
        fundus_img = fundus_img.cuda()
        oct_img = oct_img.cuda()

        if self.mode == 'test':
            return fundus_img, oct_img, real_index
        if self.mode == "train":
            label = torch.argmax(torch.tensor(label).cuda(), dim=0)  # 将 label 转换为 Tensor，并移到 GPU 上，索引
            #print(label)
            return fundus_img, oct_img, label

    def __len__(self):
        return len(self.file_list)


img_train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(
        image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30)
])

oct_train_transforms = transforms.Compose([
    transforms.CenterCrop(tuple(oct_img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

img_val_transforms = transforms.Compose([
    transforms.CenterCrop([image_size, image_size])
])

oct_val_transforms = transforms.Compose([
    transforms.CenterCrop(tuple(oct_img_size))
])

train_dataset = GAMMA_sub1_dataset(dataset_root=trainset_root,
                                   img_transforms=img_train_transforms,
                                   oct_transforms=oct_train_transforms,
                                   #filelists=train_filelists,
                                   label_file='/root/autodl-tmp/SupContrast/datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx')
print("Length of training dataset: ", len(train_dataset))
print("Batch size: ", batchsize)

val_dataset = GAMMA_sub1_dataset(dataset_root=trainset_root,
                                 img_transforms=img_val_transforms,
                                 oct_transforms=oct_val_transforms,
                                 filelists=val_filelists,
                                 label_file='/root/autodl-tmp/SupContrast/datasets/gamma/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx')
print("Length of val dataset: ", len(train_dataset))
print("Batch size: ", batchsize)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fundus_branch = resnet50(pretrained=True)
        self.oct_branch = resnet50(pretrained=True)
        #这里哦
        self.fundus_branch.fc = nn.Identity()  # remove final fc
        self.oct_branch.fc = nn.Identity()  # remove final fc
        #这里哦
        self.decision_branch = nn.Linear(512 * 1*2 , 3)  # ResNet50 use basic block, expansion = 1

        # replace first conv layer in oct_branch
        self.oct_branch.conv1 = nn.Conv2d(256, 64,
                                          kernel_size=7,
                                          stride=2,
                                          padding=3,
                                          bias=False)

    def forward(self, fundus_img, oct_img):
        #这里哦
        b1 = self.fundus_branch(fundus_img)
        b2 = self.oct_branch(oct_img)
        #这里哦
        b1 = b1.view(b1.size(0), -1)
        b2 = b2.view(b2.size(0), -1)
        #这里哦
        logit = self.decision_branch(torch.cat([b1, b2], 1))
        #logit = self.decision_branch(b1)
        #print("The logit we get is:")
        #print(logit)

        return logit


def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, log_interval, eval_interval):
    iter = 0
    print("start training...")
    model.train()
    avg_loss_list = []
    avg_kappa_list = []
    best_kappa = 0.
    while iter < iters:
        for data in train_dataloader:
            iter += 1
            if iter > iters:
                break
            fundus_imgs = data[0] / 255.0
            oct_imgs = data[1] / 255.0
            labels = data[2].long()

            logits = model(fundus_imgs, oct_imgs)
            loss = criterion(logits, labels)

            for p, l in zip(logits.argmax(1).cpu().numpy(), labels.cpu().numpy()):                avg_kappa_list.append([p, l])
            print("train时候的p和l分别为：")
            print(p)
            print(l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss_list.append(loss.item())

            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_kappa_list = np.array(avg_kappa_list)
                avg_kappa = cohen_kappa_score(avg_kappa_list[:, 0], avg_kappa_list[:, 1], weights='quadratic')
                avg_loss_list = []
                avg_kappa_list = []
                print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_kappa={:.4f}".format(iter, iters, avg_loss, avg_kappa))

            if iter % eval_interval == 0:
                avg_loss, avg_kappa = val(model, val_dataloader, criterion)
                print("[EVAL] iter={}/{} avg_loss={:.4f} kappa={:.4f}".format(iter, iters, avg_loss, avg_kappa))
                if avg_kappa >= best_kappa:
                    best_kappa = avg_kappa
                    if not os.path.exists("best_model_6"):  
                        os.makedirs("best_model_6") 
                    torch.save(model.state_dict(),
                               os.path.join("best_model_6", 'model.pth'))
                model.train()


def val(model, val_dataloader, criterion):
    model.eval()
    avg_loss_list = []
    cache = []
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for data in val_dataloader:
            fundus_imgs = data[0] / 255.0
            oct_imgs = data[1] / 255.0
            labels = data[2].long()

            logits = model(fundus_imgs, oct_imgs)
            for p, l in zip(logits.argmax(1).cpu().numpy(), labels.cpu().numpy()):  
                print("validation时候的P和L为：")
                print(p)
                print(l)              
                cache.append([p, l])
                true_labels.append(l)
                predicted_labels.append(p)

            loss = criterion(logits, labels)
            avg_loss_list.append(loss.item())
    
    cache = np.array(cache)
    kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights='quadratic')
    avg_loss = np.array(avg_loss_list).mean()
    
    
    print("The True Label is :")
    print(true_labels)
    print("The predicted label is:")
    print(predicted_labels)
    
    # Calculate F1 Score and Accuracy
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print("F1 Score: {:.4f}".format(f1))
    print("Accuracy: {:.4f}".format(accuracy))
    print("Kappa: {:.4f}".format(kappa))
    
    return avg_loss, kappa

model = Model().cuda()

if optimizer_type == "adam":
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

criterion = nn.CrossEntropyLoss()

train(model, iters, train_loader, val_loader, optimizer, criterion, log_interval=10, eval_interval=100)



best_model_path = "./best_model_6/model.pth"
model = Model().cuda()
model.load_state_dict(torch.load(best_model_path))
model.eval()

img_test_transforms = transforms.Compose([
    transforms.CenterCrop([image_size, image_size])
])

oct_test_transforms = transforms.Compose([
    transforms.CenterCrop(tuple(oct_img_size)),
])

test_dataset = GAMMA_sub1_dataset(dataset_root=test_root,
                                  img_transforms=img_test_transforms,
                                  oct_transforms=oct_test_transforms,
                                  mode='test')

#cache = []
#for fundus_img, oct_img, idx in test_dataset:
#    fundus_img = fundus_img.unsqueeze(0).cuda()
#    oct_img = oct_img.unsqueeze(0).cuda()
#
#    fundus_img = (fundus_img / 255.0).float()
#    oct_img = (oct_img / 255.0).float()

#    logits = model(fundus_img, oct_img)
#    cache.append([idx, logits.argmax(1).cpu().numpy()])

#submission_result = pd.DataFrame(cache, columns=['data', 'dense_pred'])

#submission_result['non'] = submission_result['dense_pred'].apply(lambda x: int(x[0] == 0))
#submission_result['early'] = submission_result['dense_pred'].apply(lambda x: int(x[0] == 1))
#submission_result['mid_advanced'] = submission_result['dense_pred'].apply(lambda x: int(x[0] == 2))

#submission_result[['data', 'non', 'early', 'mid_advanced']].to_csv("./submission_sub1.csv", index=False)


val2_dataset = GAMMA_sub1_dataset(dataset_root=test_root,
                                 img_transforms=img_train_transforms,
                                 oct_transforms=oct_train_transforms,
                                 mode='train',
                                 label_file='/root/autodl-tmp/SupContrast/datasets/gamma/Glaucoma_grading/Test/glaucoma_grading_testing_GT.xlsx')


test2_loader = torch.utils.data.DataLoader(
    val2_dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

avg_loss, avg_kappa2 = val(model, test2_loader, criterion)

print("The final kappa is:")
print(avg_kappa2)