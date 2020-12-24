import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder

#################################################################################
#################################################################################
#################################################################################
'''
加载预训练好的模型
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.res = torchvision.models.vgg16(pretrained=True)
        self.out1 = nn.Linear(1000, 500)
        self.out2 = nn.Linear(500, 250)
        self.out3 = nn.Linear(250, 2)
        self.dropout = nn.Dropout(0.5)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input):
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
reloadmodel = torch.load('/home/zsz2/pytorch-code/xjy/save/ceshi.pth').to(device)


'''
打印模型
'''
from torchsummary import summary
summary(reloadmodel, (3, 224, 224))


'''
搭建子模型
去除网络的最后一个全连接层
构建序列模型
'''
module = list(reloadmodel.children())[:3]
model = nn.Sequential(*module)
#################################################################################
#################################################################################
#################################################################################



#################################################################################
#################################################################################
#################################################################################
'''
加载数据以及数据预处理
'''

BATCH_SIZE = 1

'''
加载测试集
'''

test_transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
])
test = ImageFolder('/data/zsz2/leave_one_site_out/nyu_out2/test', test_transform)
test_num = len(test)
test_loader = Data.DataLoader(
    dataset=test,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

'''
加载训练集
'''
train_transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
])
train = ImageFolder('/data/zsz2/leave_one_site_out/nyu_out2/train', train_transform)
train_num = len(train)
train_loader = Data.DataLoader(
    dataset=train,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

'''
加载验证集
'''
val_transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
])
val = ImageFolder('/data/zsz2/leave_one_site_out/nyu_out2/val', val_transform)
val_num = len(val)
val_loader = Data.DataLoader(
    dataset=val,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

#################################################################################
#################################################################################
#################################################################################



#################################################################################
#################################################################################
#################################################################################
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

np.save('/home/zsz2/pytorch-code/xjy/save/train_features.npy', train_features)
np.save('/home/zsz2/pytorch-code/xjy/save/train_labels.npy', train_labels)

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

np.save('/home/zsz2/pytorch-code/xjy/save/test_features.npy', test_features)
np.save('/home/zsz2/pytorch-code/xjy/save/test_labels.npy', test_labels)

'''
保存验证集的特征和标签
'''
val_features = []
val_labels = []

model.eval()
with torch.no_grad():
    for step, (x, y) in enumerate(val_loader):
        val_data = x.to(device)
        val_label = y.to(device)
        val_pre = model(val_data)
        val_pre = val_pre.cpu().numpy()
        val_label = val_label.cpu().numpy()
        val_features.append(val_pre)
        val_labels.append(val_label)

np.save('/home/zsz2/pytorch-code/xjy/save/val_features.npy', val_features)
np.save('/home/zsz2/pytorch-code/xjy/save/val_labels.npy', val_labels)











