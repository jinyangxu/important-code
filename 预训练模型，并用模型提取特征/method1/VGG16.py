import torchvision
import torch
import  matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder


'''
设置超参数
'''
EPOCH = 5
BATCH_SIZE = 64
LR = 0.0001
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

'''
加载数据集，并对训练集做数据增强
'''

'''
train
'''
train_transform = transforms.Compose([
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(30),
         transforms.Resize((224, 224)),
         transforms.ToTensor(), 
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

model = VGG16()
model = model.to(device)

# from torchsummary import summary
# summary(model,(3,224,224))

'''
定义损失函数和优化器
'''
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(device)

'''
训练与测试
'''

def train(epoch, device, optimizer, model, train_loader, loss_func):
    model.train()
    train_average = []
    for step, (x, y) in enumerate(train_loader):
        train_data = x.to(device)
        train_labels = y.to(device)
        x_num = len(x)
        optimizer.zero_grad()
        train_pre = model(train_data)
        train_loss = loss_func(train_pre, train_labels)
        train_loss.backward()
        optimizer.step()
        train_x = torch.max(train_pre, 1)[1]  # 输出最大值与最大值的索引,[1]表示仅输出最大的索引
        train_accuracy = (train_x == train_labels).sum().item() / float(x_num)
        '''
        求每一个train epoch的平均分
        '''
        train_average.append(train_accuracy)
        train_accuracy = np.sum(train_average) / (step + 1)
        # print('train_acc:',train_accuracy)
        if (step+1) % 5 == 0:
            print("Epoch: %d/%d, Iter: %d/%d, Train_acc: %.4f, Loss: %.4f" % (
              epoch, EPOCH, step + 1, train_num // BATCH_SIZE, train_accuracy, train_loss.item()))
    Train_loss.append(train_loss.item())
    Train_acc.append(train_accuracy)

def test(device, model, test_loader, loss_func):
    model.eval()
    test_average = []
    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader):
            test_data = x.to(device)
            test_labels = y.to(device)
            x_num = len(x)
            test_pre = model(test_data)
            test_loss = loss_func(test_pre, test_labels)
            test_x = torch.max(test_pre, 1)[1]
            test_accuracy = (test_labels == test_x).sum().item() / float(x_num)
            '''
            求每一个test epoch的平均分
            '''
            test_average.append(test_accuracy)
            test_accuracy = np.sum(test_average)/(step+1)
            if (step+1) % 5 == 0:
                print("Epoch: %d/%d, Iter: %d/%d, Test_acc: %.4f, Loss: %.4f" % (epoch, EPOCH, step+1, test_num//BATCH_SIZE, test_accuracy,  test_loss.item()))

        Test_loss.append(test_loss.item())
        Test_acc.append(test_accuracy)
        return test_accuracy


if __name__ == '__main__':
    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []
    max_acc = 0
    for epoch in range(EPOCH):
        train(epoch, device, optimizer, model, train_loader, loss_func)
        test_acc = test(device, model, test_loader, loss_func)
        if test_acc > max_acc:
            max_acc = test_acc
            print('save best model')
            torch.save(model, '/home/zsz2/pytorch-code/xjy/save/5fold+emci/VGG1.pth')

'''
绘制训练曲线
'''

plt.figure(1)
plt.plot(Train_acc, label='Train_acc')
plt.plot(Test_acc, label='Test_acc')
plt.ylabel('Train_Test accurecy')
plt.xlabel('EPOCH')
plt.legend()
plt.savefig('/home/zsz2/pytorch-code/xjy/save/5fold+emci/VGG1_acc.png')


plt.figure(2)
plt.plot(Test_loss, label='Test_loss')
plt.plot(Train_loss, label='Train_loss')
plt.ylabel('Train_Test loss')
plt.xlabel('EPOCH')
plt.legend()
plt.savefig('/home/zsz2/pytorch-code/xjy/save/5fold+emci/VGG1_loss.jpg')

    

