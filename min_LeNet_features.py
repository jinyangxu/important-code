import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

'''
加载预训练好的模型
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''
定义LeNet网络模型并加载预训练权重
'''


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

'''
加载GPU上预训练好的模型
'''
reloadmodel = torch.load('/home/zsz2/pytorch-code/xjy/save/nyu.pth').to(device)


'''
搭建子模型
去除网络的最后一个全连接层
构建序列模型
'''
# module = list(reloadmodel.children())[:4]
# model = nn.Sequential(*module)
# model.load_state_dict(reloadmodel)


class min_LeNet5(nn.Module):
    def __init__(self):
        super(min_LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        #y = self.relu4(y)
        #y = self.fc3(y)
        #y = self.relu5(y)
        return y


model = min_LeNet5()
model.load_state_dict(reloadmodel.state_dict(), strict=False)

model.to(device)


'''
加载数据集，并对训练集做数据增强
'''
BATCH_SIZE = 1
'''
train
'''
train_transform = transforms.Compose([
         #transforms.Resize((32, 32)),
         transforms.ToTensor(),
])
train = ImageFolder('/data/zsz2/ceshi/train', train_transform)
train_num = len(train)
train_loader = Data.DataLoader(
    dataset=train,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)
'''
test
'''
test_transform = transforms.Compose([
         #transforms.Resize((32, 32)),
         transforms.ToTensor(),
])
test = ImageFolder('/data/zsz2/ceshi/test', test_transform)
test_num = len(test)
test_loader = Data.DataLoader(
    dataset=test,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)


'''
使用重构的子模型分别提取训练集，测试集，验证集的特征并保存
'''


'''
保存训练集的特征和标签
'''
train_features = []
train_labels = []

model.eval()
with torch.no_grad():
    for step, (x, y) in enumerate(train_loader):
        train_data = x.to(device)
        train_label = y.to(device)
        train_pre = model(train_data)
        train_pre = train_pre.cpu().numpy()
        train_label = train_label.cpu().numpy()
        train_features.append(train_pre)
        train_labels.append(train_label)

np.save('/home/zsz2/pytorch-code/xjy/save/nyu_train_features.npy', train_features)
np.save('/home/zsz2/pytorch-code/xjy/save/nyu_train_labels.npy', train_labels)

'''
保存测试集的特征和标签
'''
test_features = []
test_labels = []

model.eval()
with torch.no_grad():
    for step, (x, y) in enumerate(test_loader):
        test_data = x.to(device)
        test_label = y.to(device)
        test_pre = model(test_data)
        test_pre = test_pre.cpu().numpy()
        test_label = test_label.cpu().numpy()
        test_features.append(test_pre)
        test_labels.append(test_label)

np.save('/home/zsz2/pytorch-code/xjy/save/nuy_test_features.npy', test_features)
np.save('/home/zsz2/pytorch-code/xjy/save/nyu_test_labels.npy', test_labels)


