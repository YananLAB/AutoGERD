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
                    filename='ResNet50_2_info.log',
                    filemode='a',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # Define the first fully connected layer, compressing the number of input channels to in_channels // reduction.
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        # Define the second fully connected layer to restore the number of channels to the original input channel count.
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        # Define the Sigmoid activation function used to generate attention weights.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Get the batch_size and channels of the input tensor.
        batch_size, channels, _, _ = x.size()
        # Perform global average pooling on the spatial dimensions (height and width) of the input tensor.
        squeeze = torch.mean(x, dim=(2, 3))
        # Channel compression is performed through the first fully connected layer.
        squeeze = self.fc1(squeeze)
        # Channel expansion is carried out through the ReLU activation function and the second fully connected layer.
        squeeze = self.fc2(F.relu(squeeze))
        # Use Sigmoid to generate attention weights and adjust the shape to match the dimensions of the input tensor.
        attention = self.sigmoid(squeeze).view(batch_size, channels, 1, 1)
        # Apply the attention weights to the input tensor for channel weighting.
        return x * attention


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel attention module (SEBlock), used to learn attention weights between channels.
        self.channel_attention = SEBlock(in_channels, reduction)
        # Spatial attention module, using 1x1 convolution kernels to learn spatial attention weights.
        self.spatial_attention = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Apply the channel attention module to perform channel weighting on the input features.
        x = self.channel_attention(x)
        # Calculate the average of the input features along the channel dimension.
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Calculate the maximum value of the input features along the channel dimension.
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate the average value and the maximum value together.
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        # Learn spatial attention weights through the spatial attention module.
        spatial_out = self.spatial_attention(spatial_out)
        # Generate spatial attention weights using the Sigmoid activation function.
        spatial_attention = torch.sigmoid(spatial_out)
        # Apply the spatial attention weights to the input features to perform spatial weighting.
        return x * spatial_attention


class DualAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(DualAttentionBlock, self).__init__()
        # Channel attention module (SEBlock), used to learn the attention weights between channels.
        self.channel_attention = SEBlock(in_channels, reduction)
        # The Spatial Attention Module (CBAM) is used to learn spatial attention weights.
        self.spatial_attention = CBAM(in_channels, reduction)

    def forward(self, x):
        # Apply the channel attention module to perform channel weighting on the input features.
        x = self.channel_attention(x)
        # Apply the spatial attention module to weight the input features spatially.
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

class ResNetWithAttention(nn.Module):
    def __init__(self, attention_cls, pretrained=True):
        super(ResNetWithAttention, self).__init__()

        # Use the pre-trained ResNet50
        self.base_model = models.resnet50(pretrained=pretrained)

        # Create attention module
        # self.attention_layer1 = attention_cls(64)
        # self.attention_layer2 = attention_cls(2048)

    def forward(self, x):
        # The forward propagation process of ResNet50
        x = self.base_model.conv1(x)  # Initial convolution layer
        x = self.base_model.bn1(x)  # Batch normalization
        x = self.base_model.relu(x)  # Activation function

        # First attention module: after the first layer convolution
        # x = self.attention_layer1(x)

        # Max Pooling Layer
        x = self.base_model.maxpool(x)

        # Residual layer of ResNet
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        # Second attention module: after the last convolution layer
        # x = self.attention_layer2(x)

        # Average pooling
        x = self.base_model.avgpool(x)

        # Flatten and pass through the fully connected layer.
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        
        return x



# read data
transforms = transforms.Compose([
    transforms.Resize([224,224]),    # Scale the shorter side of the image to 224 while maintaining the aspect ratio.
    transforms.RandomHorizontalFlip(),   # Randomly flip the image
    transforms.ToTensor()          # Normalize the images and convert the data into Tensor type.
])


class MyDataset(Dataset):
    def __init__(self, img_path, transform=transforms):
        super(MyDataset, self).__init__()

        
        self.img = img_path.iloc[:,0].values.tolist()
        self.label = img_path.iloc[:,1].values.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]

        img = Image.open(img).convert('RGB')

        # img is of type PIL.Image and label is of type str.

        if self.transform is not None:
            img = self.transform(img)

        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)

        return img, label
    
train_image_paths = pd.read_csv('../output/temp_data/test.csv',header=0,index_col=0)
valid_image_paths = pd.read_csv('../output/temp_data/validation.csv',header=0,index_col=0)
train_dataset = MyDataset(train_image_paths)
valid_dataset = MyDataset(valid_image_paths)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=10, shuffle=True,num_workers=8, pin_memory=True, drop_last=True)


def train(net, train_iter, criterion, optimizer, num_epochs, device, num_print, lr_scheduler=None, test_iter=None):
    net.train()
    record_train = list()
    record_test = list()

    for epoch in range(num_epochs):
        print("========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))
        logging.info("========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))
        total, correct, train_loss = 0, 0, 0
        start = time.time()

        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            output= net(X)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
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

        if lr_scheduler is not None:
            lr_scheduler.step()

        print("--- cost time: {:.4f}s ---".format(time.time() - start))
        logging.info("--- cost time: {:.4f}s ---".format(time.time() - start))

        if test_iter is not None:
            record_test.append(test(net, test_iter, criterion, device))
        record_train.append(train_acc)

    return record_train, record_test

def test(net, test_iter, criterion, device):
    total, correct = 0, 0
    net.eval()
    pred_label = []
    true_label = []
    pred_proba = []  # Used to store the predicted probabilities of the positive class.

    with torch.no_grad():
        print("*************** test ***************")
        logging.info("*************** test ***************")
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)

            output = net(X)
            probs = F.softmax(output, dim=1)  # Convert to probability

            # Obtain predicted category and predicted probability
            predicted_classes = torch.argmax(probs, dim=1)

            # Obtain the predicted probability of the positive class (Category 1) for ROC.
            pos_probs = probs[:, 1].cpu().numpy()

            loss = criterion(output, y)

            total += y.size(0)
            correct += (predicted_classes == y).sum().item()

            # Save results
            pred_label.extend(predicted_classes.cpu().numpy())
            pred_proba.extend(pos_probs)          # Save the positive class probability
            true_label.extend(y.cpu().numpy())

    # Calculate ROC curve and AUC
    if len(set(true_label)) > 1:  # The ROC can only be calculated when both positive and negative samples are present.
        fpr, tpr, _ = roc_curve(true_label, pred_proba)
        roc_auc = auc(fpr, tpr)
        print("roc_auc: {:.4f}".format(roc_auc))
    else:
        roc_auc = None
        print("Only one class present, cannot compute ROC AUC.")

    # Calculate accuracy
    test_acc = 100.0 * correct / total
    print("test_loss: {:.3f} | test_acc: {:6.3f}%".format(loss.item(), test_acc))
    logging.info("test_loss: {:.3f} | test_acc: {:6.3f}%".format(loss.item(), test_acc))
    print("************************************\n")
    logging.info("************************************\n")

    net.train()

    return pred_label, true_label, pred_proba

def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
import matplotlib.pyplot as plt


def learning_curve(record_train, record_test=None):
    plt.style.use("ggplot")

    plt.plot(range(1, len(record_train) + 1), record_train, label="train acc")
    if record_test is not None:
        plt.plot(range(1, len(record_test) + 1), record_test, label="test acc")

    plt.legend(loc=4)
    plt.title("learning curve")
    plt.xticks(range(0, len(record_train) + 1, 5))
    plt.yticks(range(0, 101, 5))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig('ResNet50_attetion.png')








BATCH_SIZE = 32
NUM_EPOCHS = 250
NUM_CLASSES = 8
LEARNING_RATE = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_PRINT = 100
def main():
    device = torch.device("cuda:0")
    net = models.resnet50(pretrained=True)
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

    record_train, record_test = train(net, train_data_loader, criterion, optimizer, NUM_EPOCHS, device, NUM_PRINT, lr_scheduler, valid_data_loader)
    pred_label,true_label,pred_proba = test(net,valid_data_loader, criterion, device)
    
if __name__ == '__main__':
    main()
    

