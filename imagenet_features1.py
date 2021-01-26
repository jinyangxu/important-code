import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model = model.to(device)

from torchsummary import summary
summary(model, (3, 224, 224))


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
test = ImageFolder('/data/zsz2/fold/glass_data1/usm/test', test_transform)
test_num = len(test)
test_loader = Data.DataLoader(
    dataset=test,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=False,               # 要设置为False
    num_workers=2,              # subprocesses for loading data
)

'''
加载训练集
'''
train_transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
])
train = ImageFolder('/data/zsz2/fold/glass_data1/usm/train', train_transform)
train_num = len(train)
train_loader = Data.DataLoader(
    dataset=train,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=False,               # 要设置为False
    num_workers=2,              # subprocesses for loading data
)


'''
保存训练集的特征和标签
'''
train_features = []
train_labels = []
'''
保存测试集的特征和标签
'''
test_features = []
test_labels = []


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

    for step, (x, y) in enumerate(test_loader):
        test_data = x.to(device)
        test_label = y.to(device)
        test_pre = model(test_data)
        test_pre = test_pre.cpu().numpy()
        test_label = test_label.cpu().numpy()
        test_features.append(test_pre)
        test_labels.append(test_label)

np.save('/home/zsz2/pytorch-code/xjy/save/fold/usm/vgg2_train_usmfeatures.npy', train_features)
np.save('/home/zsz2/pytorch-code/xjy/save/fold/usm/vgg2_train_usmlabels.npy', train_labels)


np.save('/home/zsz2/pytorch-code/xjy/save/fold/usm/vgg2_test_usmfeatures.npy', test_features)
np.save('/home/zsz2/pytorch-code/xjy/save/fold/usm/vgg2_test_usmlabels.npy', test_labels)

