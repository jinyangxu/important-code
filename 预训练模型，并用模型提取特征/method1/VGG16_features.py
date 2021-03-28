import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from tqdm import tqdm

'''
加载预训练好的模型
'''
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

'''
定义VGG16网络模型并加载预训练权重
'''

'''
构建模型
'''
class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.res = torchvision.models.vgg16(pretrained=True)
        self.out1 = nn.Linear(1000, 500)
        self.out2 = nn.Linear(500, 250)
        self.out3 = nn.Linear(250, 2)
        self.dropout = nn.Dropout(0.5)
        #self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        x = self.res(input)
        x = self.out1(x)
        x = self.dropout(x)
        x = self.out2(x)
        x = self.dropout(x)
        x = self.out3(x)
       # x = self.sigmoid(x)

        return x
'''
加载GPU上预训练好的模型
'''
reloadmodel = torch.load('/home/zsz2/pytorch-code/xjy/save/5fold+emci/VGG1.pth').to(device)


'''
搭建子模型
去除网络的最后一个全连接层
构建序列模型
'''
class min_VGG16(nn.Module):
    def __init__(self):
        super(min_VGG16, self).__init__()
        self.res = torchvision.models.vgg16(pretrained=True)
        self.out1 = nn.Linear(1000, 500)
        self.out2 = nn.Linear(500, 250)
        self.out3 = nn.Linear(250, 2)
        self.dropout = nn.Dropout(0.5)
        #self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        x = self.res(input)
        x = self.out1(x)
        x = self.dropout(x)
        x = self.out2(x)
       # x = self.dropout(x)
       # x = self.out3(x)
       # x = self.sigmoid(x)
        return x


model = min_VGG16()
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
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
       #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train = ImageFolder('/data/zsz2/emci/data5/train', train_transform)
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
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
])
test = ImageFolder('/data/zsz2/emci/data5/test', test_transform)
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
    for step, (x, y) in tqdm(enumerate(train_loader)):
        train_data = x.to(device)
        train_label = y.to(device)
        train_pre = model(train_data)
        train_pre = train_pre.cpu().numpy()
        train_label = train_label.cpu().numpy()
        train_features.append(train_pre)
        train_labels.append(train_label)

np.save('/home/zsz2/pytorch-code/xjy/save/5fold+emci/vgg1_train_features.npy', train_features)
np.save('/home/zsz2/pytorch-code/xjy/save/5fold+emci/vgg1_train_labels.npy', train_labels)

'''
保存测试集的特征和标签
'''
test_features = []
test_labels = []

model.eval()
with torch.no_grad():
    for step, (x, y) in tqdm(enumerate(test_loader)):
        test_data = x.to(device)
        test_label = y.to(device)
        test_pre = model(test_data)
        test_pre = test_pre.cpu().numpy()
        test_label = test_label.cpu().numpy()
        test_features.append(test_pre)
        test_labels.append(test_label)

np.save('/home/zsz2/pytorch-code/xjy/save/5fold+emci/vgg1_test_features.npy', test_features)
np.save('/home/zsz2/pytorch-code/xjy/save/5fold+emci/vgg1_test_labels.npy', test_labels)






