import torchvision
import torch
import  matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

###设置超参数########
EPOCH = 5
BATCH_SIZE = 64
LR = 0.0005
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

####添加高斯噪声##########
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

###加载数据集，并对训练集做数据增强
######train#########
train_transform  = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=0.5),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
        # transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
         transforms.Resize((224, 224)),
        # AddGaussianNoise(),
         transforms.ToTensor(), 
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 0.3), value=0, inplace=True),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train = ImageFolder('/data/zsz2/emci/data5/train',train_transform)
train_num = len(train)
train_loader = Data.DataLoader(
    dataset=train,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)
######test##############
test_transform  = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
])

test = ImageFolder('/data/zsz2/emci/data5/test',test_transform)
test_num = len(test)
test_loader = Data.DataLoader(
    dataset=test,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)


##########构建模型############
class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.res = torchvision.models.vgg16(pretrained=True)
        self.out1 = nn.Linear(1000,512)
        self.out2 = nn.Linear(512,256)
        self.out3 = nn.Linear(256,2)
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.4)
        #self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        x = self.res(input)
        x = self.out1(x)
        x = self.dropout1(x)
        x = self.out2(x)
        x = self.dropout2(x)
        x = self.out3(x)
       # x = self.sigmoid(x)

        return x

model = Resnet()
model = model.to(device)

# from torchsummary import summary
# summary(model,(3,224,224))

########定义损失函数和优化器################
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(device)

#########训练与测试######################


def train(epoch,device,optimizer,model,train_loader,loss_func):
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
        train_x = torch.max(train_pre,1)[1] ##输出最大值与最大值的索引,[1]表示仅输出最大的索引
        train_accuracy = (train_x == train_labels).sum().item()/float(x_num)
        ##########################求每一个train epoch的平均分###############################
        train_average.append(train_accuracy)
        train_accuracy = np.sum(train_average)/(step+1)
        #print('train_acc:',train_accuracy)
        if (step+1)%5 == 0:
            print("Epoch: %d/%d, Iter: %d/%d, Train_acc: %.4f, Loss: %.4f"%(epoch, EPOCH, step+1, train_num//BATCH_SIZE, train_accuracy,  train_loss.item()))    
    Train_loss.append(train_loss.item())
    Train_acc.append(train_accuracy)        

def test(device,model,test_loader,loss_func):
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
            ##########################求每一个test epoch的平均分###############################
            test_average.append(test_accuracy)
            test_accuracy = np.sum(test_average)/(step+1)
            if (step+1)%5 == 0: 
                print("Epoch: %d/%d, Iter: %d/%d, Test_acc: %.4f, Loss: %.4f"%(epoch, EPOCH, step+1, test_num//BATCH_SIZE, test_accuracy,  test_loss.item()))
        Test_loss.append(test_loss.item())
        Test_acc.append(test_accuracy)

#Train_loss = []
#Train_acc = []
#Test_loss = []
#Test_acc = []


if __name__ =='__main__':
    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = [] 
    for epoch in range(EPOCH):
        train(epoch=epoch,device=device,optimizer=optimizer,model=model,train_loader=train_loader,loss_func=loss_func)
        test(device,model,test_loader,loss_func)
      #  Test_loss.append(test_loss)
      #  Test_acc.append(test_accuracy)
        # print('epoch',epoch,'loss',loss,'train_acc',train_accuracy,'test_accuracy',test_accuracy)
        # loss_value = loss_value.append(loss)
        # train_acc = train_acc.append(train_accuracy)
        # test_acc = test_acc.append(test_accuracy)


###绘制训练曲线

plt.figure(1)
plt.plot(Train_acc, label = 'Train_acc')
plt.plot(Test_acc, label = 'Test_acc')
plt.ylabel('Train_Test accurecy')
plt.xlabel('EPOCH')
plt.legend()
plt.savefig('/home/zsz2/pytorch-code/xjy/save/5fold+emci/VGG1_acc.png')


plt.figure(2)
plt.plot(Test_loss, label = 'Test_loss')
plt.plot(Train_loss, label = 'Train_loss')
plt.ylabel('Train_Test loss')
plt.xlabel('EPOCH')
plt.legend()
plt.savefig('/home/zsz2/pytorch-code/xjy/save/5fold+emci/VGG1_loss.jpg')

torch.save(model, '/home/zsz2/pytorch-code/xjy/save/5fold+emci/VGG1.pth')



