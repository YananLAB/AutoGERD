import torch
from torch import nn
import logging
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import os
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import argparse
import random
import shutil
import time
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
# import resnet_test

import torchvision
from torch import nn
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
# import ResNet50
# import DistributedResnet50.image_classification.resnet as nvmodels

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO,
                    filename='ResNet50_last_info.log',
                    filemode='a',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # 定义第一个全连接层，将输入通道数压缩为 in_channels // reduction
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        # 定义第二个全连接层，将通道数恢复为原始输入通道数
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        # 定义Sigmoid激活函数，用于生成注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入张量的 batch_size 和 channels
        batch_size, channels, _, _ = x.size()
        # 对输入张量的空间维度（高度和宽度）进行全局平均池化
        squeeze = torch.mean(x, dim=(2, 3))
        # 通过第一个全连接层进行通道压缩
        squeeze = self.fc1(squeeze)
        # 通过ReLU激活函数和第二个全连接层进行通道扩展
        squeeze = self.fc2(F.relu(squeeze))
        # 使用Sigmoid生成注意力权重，并调整形状以匹配输入张量的维度
        attention = self.sigmoid(squeeze).view(batch_size, channels, 1, 1)
        # 将注意力权重应用到输入张量上，进行通道加权
        return x * attention


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力模块（SEBlock），用于学习通道间的注意力权重
        self.channel_attention = SEBlock(in_channels, reduction)
        # 空间注意力模块，使用1x1卷积核学习空间注意力权重
        self.spatial_attention = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # 应用通道注意力模块，对输入特征进行通道加权
        x = self.channel_attention(x)
        # 计算输入特征在通道维度上的平均值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 计算输入特征在通道维度上的最大值
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均值和最大值拼接在一起
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        # 通过空间注意力模块学习空间注意力权重
        spatial_out = self.spatial_attention(spatial_out)
        # 使用Sigmoid激活函数生成空间注意力权重
        spatial_attention = torch.sigmoid(spatial_out)
        # 将空间注意力权重应用到输入特征上，进行空间加权
        return x * spatial_attention


class DualAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(DualAttentionBlock, self).__init__()
        # 通道注意力模块（SEBlock），用于学习通道间的注意力权重
        self.channel_attention = SEBlock(in_channels, reduction)
        # 空间注意力模块（CBAM），用于学习空间上的注意力权重
        self.spatial_attention = CBAM(in_channels, reduction)

    def forward(self, x):
        # 应用通道注意力模块，对输入特征进行通道加权
        x = self.channel_attention(x)
        # 应用空间注意力模块，对输入特征进行空间加权
        x = self.spatial_attention(x)
        return x

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention



class Model2_3(nn.Module):
    def __init__(self):
        super(Model2_3, self).__init__()
        # 减少通道数
        self.conv1 = nn.Conv2d(148, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # 增加空间维度
        self.deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(4, 3, kernel_size=4, stride=2, padding=1)
        
        # 最终调整到目标尺寸
        self.final_conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # 减少通道数
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn1(x)
        # 增加空间维度
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        
        # 最终调整到目标尺寸
        x = self.final_conv(x)
        x = self.upsample(x)
        return x



class ResNetWithAttention(nn.Module):
    def __init__(self, attention_cls, pretrained=True):
        super(ResNetWithAttention, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(148, 128, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 64x64 -> 128x128
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 128x128 -> 256x256
        
        # 减少通道数
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)
        
        # 最终调整到目标尺寸
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)
        # 使用预训练的 ResNet50
        self.base_model = models.resnet50(pretrained=pretrained)

        # 创建注意力模块
        # self.attention_layer1 = attention_cls(64)  # 第一层卷积后
        self.attention_layer2 = attention_cls(2048)  # 最后一层卷积后

    def forward(self, x,img,return_probs=True):
        # ResNet50的前向传播过程
        x = F.relu(self.deconv1(x))  # [148, 32, 32] -> [128, 64, 64]
        x = F.relu(self.deconv2(x))  # [128, 64, 64] -> [64, 128, 128]
        x = F.relu(self.deconv3(x))  # [64, 128, 128] -> [32, 256, 256]
        
        # 减少通道数
        x = F.relu(self.conv1(x))  # [32, 256, 256] -> [16, 256, 256]
        x = F.relu(self.conv2(x))  # [16, 256, 256] -> [8, 256, 256]
        x = self.conv3(x)          # [8, 256, 256] -> [4, 256, 256]
        
        # 最终调整到目标尺寸
        x = self.upsample(x) 
        x = torch.cat((x,img),dim=1)
        x = F.relu(self.conv4(x)) 
        x = self.base_model.conv1(x)  # 初始卷积层
        x = self.base_model.bn1(x)  # 批归一化
        x = self.base_model.relu(x)  # 激活函数

        

        # 最大池化层
        x = self.base_model.maxpool(x)

        # ResNet的残差层
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        # 第二个注意力模块：最后一层卷积后
        x = self.attention_layer2(x)

        # 平均池化
        x = self.base_model.avgpool(x)

        # 展平并通过全连接层
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        
        if return_probs:
            probs = F.softmax(x,dim=1)
            return x,probs
        return x



# 数据读取
transforms = transforms.Compose([
    transforms.Resize([224,224]),    # 将图片短边缩放至224，长宽比保持不变：
    transforms.RandomHorizontalFlip(),   #将图片随机翻转
    transforms.ToTensor()          #把图片进行归一化，并把数据转换成Tensor类型
])

corr = pd.read_csv('./corr.csv',header=0,index_col=0).values

class MyDataset(Dataset):
    def __init__(self, img_path, transform=transforms):
        super(MyDataset, self).__init__()

        
        self.img = img_path.loc[:,'path'].values.tolist()
        self.label = img_path.loc[:,'label'].values.tolist()
        self.date = img_path.drop(columns=['path','label']).values.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]
        data = self.date[item]
        xl = data*corr*data
        # logging.info("path:"+img)
        # xl = torch.from_numpy(np.pad(xl,32*3,'constant'))
        xl = torch.from_numpy(xl).float()
        xl = xl.reshape(1, 32,32)
        img = Image.open(img).convert('RGB')

        # 此时img是PIL.Image类型   label是str类型

        if self.transform is not None:
            img = self.transform(img)
        for i in range(0,7):
            for j in range(0,7):
                xl = torch.cat((xl,img[:,i*32:(i+1)*32,j*32:(j+1)*32]), dim=0)
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        # img = torch.cat((img, xl), dim=0)
        xl.reshape(1,148,32,32)
        # logging.info("size:"+str(xl.size()))

        return xl, img, label
    
# train_image_paths = pd.read_csv('./train_concat.csv',header=0,index_col=0)
# valid_image_paths = pd.read_csv('./test_concat.csv',header=0,index_col=0)
train_image_paths = pd.read_csv('../output/temp_data/train.csv',header=0,index_col=0)
valid_image_paths = pd.read_csv('../output/temp_data/train.csv',header=0,index_col=0)
# valid_image_paths = valid_image_paths.sample(frac=1).reset_index(drop=True)
# valid_image_paths = valid_image_paths.iloc[:229,:]
# valid_image_paths = valid_image_paths.iloc[0:6,:]
train_image_paths.reset_index(drop=True,inplace=True)
valid_image_paths.reset_index(drop=True,inplace=True)
train_dataset = MyDataset(train_image_paths)
valid_dataset = MyDataset(valid_image_paths)
# train_data_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
# valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=4, shuffle=True,num_workers=8, pin_memory=True, drop_last=True)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=10, shuffle=True,num_workers=8, pin_memory=True, drop_last=True)

def train(net, train_iter, criterion, optimizer, num_epochs, device, num_print, lr_scheduler=None, test_iter=None):
    net.train()
    record_train = list()
    record_test = list()
    #oyz
    train_AUC = list()

    for epoch in range(num_epochs):
        print("========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))
        logging.info("========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))
        total, correct, train_loss = 0, 0, 0
        start = time.time()
        #oyz
        pred_label = list()
        true_label = list()
        pred_proba = list()

        for i, (X,img, y) in enumerate(train_iter):
            # print(X.size())
            X, y ,img= X.to(device), y.to(device), img.to(device)
            output,proba= net(X,img)
            # for o in range(10):
            #     proba_temp = proba[o].reshape(1,1000)
            #     predicted_class = torch.argmax(proba_temp,dim=1).item()
            #     confidence = proba_temp[0,predicted_class].item()
            #     pred_proba.append(confidence)
            probs = F.softmax(output, dim=1)
            
            # 获取预测类别和预测概率
            predicted_classes = torch.argmax(probs, dim=1)

            # 获取正类（类别1）的预测概率用于 ROC
            pos_probs = probs[:, 1].detach().cpu().numpy()
            
            pred_label.extend(predicted_classes.cpu().numpy())
            pred_proba.extend(pos_probs)          # 保存正类概率
            true_label.extend(y.cpu().numpy())
            
            # predicted_class = torch.argmax(proba,dim=1).item()
            # confidence = proba[0,predicted_class].item()
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()
            train_acc = 100.0 * correct / total

            if (i + 1) % num_print == 0:
                print("step: [{}/{}], train_loss: {:.3f} | train_acc: {:6.3f}% | lr: {:.6f}" \
                    .format(i + 1, len(train_iter), 
                            train_loss, 
                            train_acc, get_cur_lr(optimizer)))
                logging.info("step: [{}/{}], train_loss: {:.3f} | train_acc: {:6.3f}% | lr: {:.6f}" \
                    .format(i + 1, len(train_iter), 
                            train_loss, 
                            train_acc, get_cur_lr(optimizer)))
            #oyz
            # pred_label+=list(output.argmax(dim=1).cpu().numpy())
            # true_label+=list(y.cpu().numpy())
        epoch_auc = roc_auc_score(true_label,pred_proba)
        train_AUC.append(epoch_auc*100)

        if lr_scheduler is not None:
            lr_scheduler.step()

        print("--- AUC: {:.8f} ---".format(epoch_auc))
        print("--- cost time: {:.4f}s ---".format(time.time() - start))
        logging.info("--- cost time: {:.4f}s ---".format(time.time() - start))

        if test_iter is not None:
            # record_test.append(test(net, test_iter, criterion, device))
            pass
        record_train.append(train_acc)

    return record_train, record_test,train_AUC


def test(net, test_iter, criterion, device):
    total, correct = 0, 0
    net.eval()
    pred_label = []
    true_label = []
    pred_proba = []  # This will store the probability of the positive class for each sample.

    with torch.no_grad():
        print("*************** test ***************")
        for X, img, y in test_iter:
            X, y, img = X.to(device), y.to(device), img.to(device)

            output, probs = net(X, img)
            
            # For binary classification, typically probs[:, 1] gives the prob. of the positive class.
            # If your task is multi-class and you want to compute ROC for one-vs-rest, you'll need a different approach.
            predicted_classes = torch.argmax(probs, dim=1)
            confidence_for_predicted_class = probs[torch.arange(probs.size(0)), predicted_classes].cpu().numpy()
            
            loss = criterion(output, y)
            total += y.size(0)
            correct += (predicted_classes == y).sum().item()

            # Update lists
            pred_label += list(predicted_classes.cpu().numpy())
            pred_proba += list(probs[:, 1].cpu().numpy())  # Assuming binary classification, get the prob. of the positive class
            true_label += list(y.cpu().numpy())

    fpr, tpr, _ = roc_curve(true_label, pred_proba)
    roc_auc = auc(fpr, tpr)
    print("roc_auc:{}".format(roc_auc))

    test_acc = 100.0 * correct / total

    print("test_loss: {:.3f} | test_acc: {:6.3f}%".format(loss.item(), test_acc))
    print("************************************\n")

    net.train()

    return pred_label, true_label, pred_proba

def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
import matplotlib.pyplot as plt


def learning_curve(record_train, record_test=None):
    pd.DataFrame(record_train,columns=['acc']).to_csv('../output/temp_data/record_train.csv')
    plt.clf()
    plt.style.use("ggplot")

    plt.plot(range(1, len(record_train) + 1), record_train, label="train acc")

    plt.legend(loc=4)
    plt.title("learning curve")
    plt.xticks(range(0, len(record_train) + 1, 10))
    plt.yticks(range(0, 101, 5))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig('Train_Epoch_ACC.png')
    
def plot_AUC(train_AUC, record_test=None):
    pd.DataFrame(train_AUC,columns=['auc']).to_csv('../output/temp_data/train_AUC.csv')
    plt.clf()
    plt.style.use("ggplot")

    plt.plot(range(1, len(train_AUC) + 1), train_AUC, label="train AUC")

    plt.legend(loc=4)
    plt.title("Epoch AUC")
    plt.xticks(range(0, len(train_AUC) + 1, 10))
    plt.yticks(range(0, 101, 5))
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.savefig('Train_Epoch_AUC.png')








BATCH_SIZE = 32
NUM_EPOCHS = 50
# '''
# 训练到50轮已经收敛
# '''
# NUM_EPOCHS = 50
# NUM_EPOCHS = 2
NUM_CLASSES = 8
LEARNING_RATE = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_PRINT = 100
def main():
    device = torch.device("cuda:0")
    attention_cls = DualAttentionBlock  # 可以替换为其他类型的注意力模块
    net = ResNetWithAttention(attention_cls)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        net.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    record_train, record_test, train_AUC = train(net, train_data_loader, criterion, optimizer, NUM_EPOCHS, device, NUM_PRINT, lr_scheduler, valid_data_loader)
    pred_label,true_label,pred_proba = test(net,valid_data_loader, criterion, device)
    
if __name__ == '__main__':
    main()
    

