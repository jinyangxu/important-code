import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from PIL import Image


'''
设置超参数
'''
EPOCH = 5
BATCH_SIZE = 64
LR = 0.00001
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


'''
添加高斯噪声
'''
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=25.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


'''
添加椒盐噪声
'''
class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)  # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0  # 椒
        img[mask == 1] = 255  # 盐
        img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
        return img



'''
加载数据集，并对训练集做数据增强
'''

'''
train
'''
train_transform = transforms.Compose([
         #transforms.RandomHorizontalFlip(),
         #transforms.RandomRotation(30),
         #transforms.ColorJitter(brightness=0.5),
         #transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
         transforms.Resize((32, 32)),
         transforms.ToTensor(),
         #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 0.3), value=0, inplace=True),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train = ImageFolder('/data/zsz2/fold/glass_data1/sbl/train', train_transform)
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
         transforms.Resize((32, 32)),
         transforms.ToTensor(),
])
test = ImageFolder('/data/zsz2/fold/glass_data1/sbl/test', test_transform)
test_num = len(test)
test_loader = Data.DataLoader(
    dataset=test,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

'''
定义LeNet网络模型
'''

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 这里论文上写的是conv,官方教程用了线性层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = LeNet5()
model = model.to(device)
print(model)

'''
定义损失函数和优化器
'''
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()



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
        if (step+1) % 50 == 0:
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
            if (step+1) % 50 == 0:
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
            torch.save(model, '/home/zsz2/pytorch-code/xjy/save/fold/sbl/sbl.pth')

'''
绘制训练曲线
'''

plt.figure(1)
plt.plot(Train_acc, label='Train_acc')
plt.plot(Test_acc, label='Test_acc')
plt.ylabel('Train_Test accurecy')
plt.xlabel('EPOCH')
plt.legend()
plt.savefig('/home/zsz2/pytorch-code/xjy/save/fold/sbl/sbl_acc.png')


plt.figure(2)
plt.plot(Test_loss, label='Test_loss')
plt.plot(Train_loss, label='Train_loss')
plt.ylabel('Train_Test loss')
plt.xlabel('EPOCH')
plt.legend()
plt.savefig('/home/zsz2/pytorch-code/xjy/save/fold/sbl/sbl_loss.jpg')






















