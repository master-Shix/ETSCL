"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

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
    
class ResNet2(nn.Module):
    def __init__(self, block, num_blocks, in_channel=128, zero_init_residual=False):
        super(ResNet2, self).__init__()
        self.in_planes = 64
        temp=128
        self.conv1 = nn.Conv2d(temp, 64, kernel_size=3, stride=1, padding=1,
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


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet2_128(**kwargs):
    return ResNet2(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'resnet2': [resnet2_128, 2048],
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


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    
class ResNetBaseline1_(nn.Module): #用这个来建立取代encoder来更新的网络，首先试试用resnet50
    def __init__(self,name='resnet2',feat_dim=128):
        super(ResNetBaseline1_,self).__init__()
        model_fun,dim_in=model_dict[name]
        self.encoder=model_fun()
    def forward(self,x):
        feat=self.encoder(x)
        return feat
    
class ResNetBaseline2_(nn.Module): #同一个网络设置两个先，怕以后改变
    def __init__(self,name='resnet50',feat_dim=128):
        super(ResNetBaseline2_,self).__init__()
        model_fun,dim_in=model_dict[name]
        self.encoder=model_fun()
    def forward(self,x):
        feat=self.encoder(x)
        return feat
    
class MLPBaseline_(nn.Module): #同一个网络设置两个先，怕以后改变
    def __init__(self,name='resnet50',feat_dim=128):
        super(MLPBaseline_,self).__init__()
        model_fun,dim_in=model_dict[name]
        dim_in=dim_in*2
        self.MLPtrain = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, 3)
            )
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        feat=self.MLPtrain(x)
        return feat
        

class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

class MultiLinearClassfier(nn.Module):
    def __init__(self, feat_dim=128, num_classes=3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3*feat_dim, 10)
        self.fc2 = nn.Linear(10, num_classes)
        
    def forward(self, feature_1, feature_2, feature_3):
        feature = torch.concat([feature_1, feature_2, feature_3], dim=1)
        return self.fc2(self.fc1(feature))    
    
class MultiLinearClassfier2(nn.Module):
    def __init__(self, feat_dim=128, num_classes=3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2*feat_dim, 10)
        self.fc2 = nn.Linear(10, num_classes)
        
    def forward(self, feature_1, feature_2):
        feature = torch.concat([feature_1, feature_2], dim=1)
        return self.fc2(self.fc1(feature))


class MultiLinearClassfier3(nn.Module):
    def __init__(self, feat_dim=2048, num_classes=3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3 * feat_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, feature_1, feature_2,feature_3):
        feature = torch.concat([feature_1, feature_2,feature_3], dim=1)
        return self.fc2(self.fc1(feature))
class DSClassifier(nn.Module):
    """Dempster-Shafer classifier"""
    def __init__(self, feat_dim=128, num_classes=3):
        super(DSClassifier, self).__init__()
        self.feat_dim = feat_dim
        self.num_claasses = num_classes
        self.fc1 = nn.Linear(feat_dim, num_classes)
        self.fc2 = nn.Linear(feat_dim, num_classes)
        self.fc3 = nn.Linear(feat_dim, num_classes)
        
    def DS_Combin_two(self, alpha1, alpha2):
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.num_claasses / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.num_claasses, 1), b[1].view(-1, 1, self.num_claasses))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.num_claasses / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a
    
    def forward(self, features_1, features_2, features_3):
        features_1 = self.fc1(features_1)
        features_2 = self.fc2(features_2)
        features_3 = self.fc3(features_3)
        
        evidence_1, evidence_2, evidence_3 = F.softplus(features_1), F.softplus(features_2), F.softplus(features_3)
        dirichlet_1, dirichlet_2, dirichlet_3 = evidence_1+1, evidence_2+1, evidence_3+1
        dirichlet_combined = self.DS_Combin_two(self.DS_Combin_two(dirichlet_1, dirichlet_2), dirichlet_3)
        
        return dirichlet_1, dirichlet_2, dirichlet_3, dirichlet_combined
    
    
class myModel_oct(nn.Module):
    def __init__(self, opt):
        super(myModel_oct, self).__init__()
        self.encoder = resnet50(in_channel=256)
        self.encoder.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
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


class myModel_fundus(nn.Module):
    def __init__(self, opt):
        super(myModel_fundus, self).__init__()
        self.encoder = timm.create_model('resnet50', pretrained=False, num_classes=3)
        self.encoder.fc = nn.Identity()  # 将最后的fc层替换为Identity，仅传递特征
        #self.encoder = resnet50()
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
        )

    def forward(self, data):
        feature = self.encoder(data)
        feature = torch.nn.functional.normalize(self.head(feature), dim=1)

        return feature
    
class myModel_vessel(nn.Module):
    def __init__(self, opt):
        super(myModel_vessel, self).__init__()
        self.encoder = timm.create_model('resnet50', pretrained=False, num_classes=3)
        self.encoder.fc = nn.Identity()  # 将最后的fc层替换为Identity，仅传递特征
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
        )

    def forward(self, data):
        feature = self.encoder(data)
        feature = torch.nn.functional.normalize(self.head(feature), dim=1)

        return feature