import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from nilearn.connectome import ConnectivityMeasure
from nilearn.connectome import sym_matrix_to_vec


'''
加载提取出来的特征数据集
'''

'''
加载caltech数据
'''
caltech_train_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\caltech\\caltech_train_features.npy')
caltech_train_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\caltech\\caltech_train_labels.npy')
caltech_train_features = np.squeeze(caltech_train_features)
#print("caltech_train_features.shape:", caltech_train_features.shape)
#print("caltech_train_labels.shape", caltech_train_labels.shape)

caltech_test_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\caltech\\caltech_test_features.npy')
caltech_test_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\caltech\\caltech_test_labels.npy')
caltech_test_features = np.squeeze(caltech_test_features)
#print("caltech_test_features.shape:", caltech_test_features.shape)
#print("caltech_test_labels.shape", caltech_test_labels.shape)


#caltech构建皮尔逊矩阵
caltech_addfeatures_train = []
caltech_addfeatures_test = []

for i in range(30):
    min_train = caltech_train_features[i * 146:(i + 1) * 146]
    caltech_addfeatures_train.append(min_train)

for i in range(7):
    min_test = caltech_test_features[i * 146:(i + 1) * 146]
    caltech_addfeatures_test.append(min_test)

caltech_addfeatures_train = np.squeeze(np.array(caltech_addfeatures_train))
caltech_addfeatures_test = np.squeeze(np.array(caltech_addfeatures_test))

caltech_addtrain_labels = np.array([0] * 15 + [1] * 15)
caltech_addtest_labels = np.array([0] * 4 + [1] * 3)


conn_est = ConnectivityMeasure(kind='tangent')
caltech_addfeatures_train = conn_est.fit_transform(caltech_addfeatures_train)
caltech_addfeatures_train = sym_matrix_to_vec(caltech_addfeatures_train)

caltech_addfeatures_test = conn_est.fit_transform(caltech_addfeatures_test)
caltech_addfeatures_test = sym_matrix_to_vec(caltech_addfeatures_test)


'''
加载leuven数据
'''
leuven_train_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\leuven\\leuven_train_features.npy')
leuven_train_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\leuven\\leuven_train_labels.npy')
leuven_train_features = np.squeeze(leuven_train_features)
#print("leuven_train_features.shape:", leuven_train_features.shape)
#print("leuven_train_labels.shape", leuven_train_labels.shape)

leuven_test_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\leuven\\leuven_test_features.npy')
leuven_test_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\leuven\\leuven_test_labels.npy')
leuven_test_features = np.squeeze(leuven_test_features)
#print("leuven_test_features.shape:", leuven_test_features.shape)
#print("leuven_test_labels.shape", leuven_test_labels.shape)

#leuven构建皮尔逊矩阵
leuven_addfeatures_train = []
leuven_addfeatures_test = []

for i in range(52):
    min_train = leuven_train_features[i * 246:(i + 1) * 246]
    leuven_addfeatures_train.append(min_train)

for i in range(11):
    min_test = leuven_test_features[i * 246:(i + 1) * 246]
    leuven_addfeatures_test.append(min_test)

leuven_addfeatures_train = np.squeeze(np.array(leuven_addfeatures_train))
leuven_addfeatures_test = np.squeeze(np.array(leuven_addfeatures_test))
leuven_addtrain_labels = np.array([0] * 24 + [1] * 28)
leuven_addtest_labels = np.array([0] * 5 + [1] * 6)


conn_est = ConnectivityMeasure(kind='tangent')
leuven_addfeatures_train = conn_est.fit_transform(leuven_addfeatures_train)
leuven_addfeatures_train = sym_matrix_to_vec(leuven_addfeatures_train)

leuven_addfeatures_test = conn_est.fit_transform(leuven_addfeatures_test)
leuven_addfeatures_test = sym_matrix_to_vec(leuven_addfeatures_test)

'''
加载nyu数据
'''
nyu_train_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\nyu\\nyu_train_features.npy')
nyu_train_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\nyu\\nyu_train_labels.npy')
nyu_train_features = np.squeeze(nyu_train_features)
# print("ohsu_train_features.shape:", nyu_train_features.shape)
# print("ohsu_train_labels.shape", nyu_train_labels.shape)

nyu_test_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\nyu\\nyu_test_features.npy')
nyu_test_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\nyu\\nyu_test_labels.npy')
nyu_test_features = np.squeeze(nyu_test_features)
# print("ohsu_test_features.shape:", nyu_test_features.shape)
# print("ohsu_test_labels.shape", nyu_test_labels.shape)

#nyu构建皮尔逊矩阵
nyu_addfeatures_train = []
nyu_addfeatures_test = []

for i in range(139):
    min_train = nyu_train_features[i * 176:(i + 1) * 176]
    nyu_addfeatures_train.append(min_train)

for i in range(35):
    min_test = nyu_test_features[i * 176:(i + 1) * 176]
    nyu_addfeatures_test.append(min_test)

nyu_addfeatures_train = np.squeeze(np.array(nyu_addfeatures_train))
nyu_addfeatures_test = np.squeeze(np.array(nyu_addfeatures_test))
nyu_addtrain_labels = np.array([0] * 60 + [1] * 79)
nyu_addtest_labels = np.array([0] * 15 + [1] * 20)


conn_est = ConnectivityMeasure(kind='tangent')
nyu_addfeatures_train = conn_est.fit_transform(nyu_addfeatures_train)
nyu_addfeatures_train = sym_matrix_to_vec(nyu_addfeatures_train)

nyu_addfeatures_test = conn_est.fit_transform(nyu_addfeatures_test)
nyu_addfeatures_test = sym_matrix_to_vec(nyu_addfeatures_test)



'''
加载ohsu数据
'''
ohsu_train_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\ohsu\\ohsu_train_features.npy')
ohsu_train_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\ohsu\\ohsu_train_labels.npy')
ohsu_train_features = np.squeeze(ohsu_train_features)
# print("ohsu_train_features.shape:", ohsu_train_features.shape)
# print("ohsu_train_labels.shape", ohsu_train_labels.shape)

ohsu_test_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\ohsu\\ohsu_test_features.npy')
ohsu_test_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\ohsu\\ohsu_test_labels.npy')
ohsu_test_features = np.squeeze(ohsu_test_features)
# print("ohsu_test_features.shape:", ohsu_test_features.shape)
# print("ohsu_test_labels.shape", ohsu_test_labels.shape)

#ohsu构建皮尔逊矩阵
ohsu_addfeatures_train = []
ohsu_addfeatures_test = []

for i in range(22):
    min_train = ohsu_train_features[i * 78:(i + 1) * 78]
    ohsu_addfeatures_train.append(min_train)

for i in range(4):
    min_test = ohsu_test_features[i * 78:(i + 1) * 78]
    ohsu_addfeatures_test.append(min_test)

ohsu_addfeatures_train = np.squeeze(np.array(ohsu_addfeatures_train))
ohsu_addfeatures_test = np.squeeze(np.array(ohsu_addfeatures_test))
ohsu_addtrain_labels = np.array([0] * 10 + [1] * 12)
ohsu_addtest_labels = np.array([0] * 2 + [1] * 2)


conn_est = ConnectivityMeasure(kind='tangent')
ohsu_addfeatures_train = conn_est.fit_transform(ohsu_addfeatures_train)
ohsu_addfeatures_train = sym_matrix_to_vec(ohsu_addfeatures_train)

ohsu_addfeatures_test = conn_est.fit_transform(ohsu_addfeatures_test)
ohsu_addfeatures_test = sym_matrix_to_vec(ohsu_addfeatures_test)


'''
加载olin数据
'''
olin_train_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\olin\\olin_train_features.npy')
olin_train_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\olin\\olin_train_labels.npy')
olin_train_features = np.squeeze(olin_train_features)
# print("olin_train_features.shape:", olin_train_features.shape)
# print("olin_train_labels.shape", olin_train_labels.shape)

olin_test_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\olin\\olin_test_features.npy')
olin_test_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\olin\\olin_test_labels.npy')
olin_test_features = np.squeeze(olin_test_features)
# print("olin_test_features.shape:", olin_test_features.shape)
# print("olin_test_labels.shape", olin_test_labels.shape)


#olin构建皮尔逊矩阵
olin_addfeatures_train = []
olin_addfeatures_test = []

for i in range(28):
    min_train = olin_train_features[i * 206:(i + 1) * 206]
    olin_addfeatures_train.append(min_train)

for i in range(6):
    min_test = olin_test_features[i * 206:(i + 1) * 206]
    olin_addfeatures_test.append(min_test)

olin_addfeatures_train = np.squeeze(np.array(olin_addfeatures_train))
olin_addfeatures_test = np.squeeze(np.array(olin_addfeatures_test))
olin_addtrain_labels = np.array([0] * 16 + [1] * 12)
olin_addtest_labels = np.array([0] * 3 + [1] * 3)


conn_est = ConnectivityMeasure(kind='tangent')
olin_addfeatures_train = conn_est.fit_transform(olin_addfeatures_train)
olin_addfeatures_train = sym_matrix_to_vec(olin_addfeatures_train)

olin_addfeatures_test = conn_est.fit_transform(olin_addfeatures_test)
olin_addfeatures_test = sym_matrix_to_vec(olin_addfeatures_test)

'''
加载pitt数据
'''
pitt_train_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\pitt\\pitt_train_features.npy')
pitt_train_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\pitt\\pitt_train_labels.npy')
pitt_train_features = np.squeeze(pitt_train_features)
# print("pitt_train_features.shape:", pitt_train_features.shape)
# print("pitt_train_labels.shape", pitt_train_labels.shape)

pitt_test_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\pitt\\pitt_test_features.npy')
pitt_test_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\pitt\\pitt_test_labels.npy')
pitt_test_features = np.squeeze(pitt_test_features)
# print("pitt_test_features.shape:", pitt_test_features.shape)
# print("pitt_test_labels.shape", pitt_test_labels.shape)

#pitt构建皮尔逊矩阵
pitt_addfeatures_train = []
pitt_addfeatures_test = []

for i in range(46):
    min_train = pitt_train_features[i * 196:(i + 1) * 196]
    pitt_addfeatures_train.append(min_train)

for i in range(10):
    min_test = pitt_test_features[i * 196:(i + 1) * 196]
    pitt_addfeatures_test.append(min_test)

pitt_addfeatures_train = np.squeeze(np.array(pitt_addfeatures_train))
pitt_addfeatures_test = np.squeeze(np.array(pitt_addfeatures_test))
pitt_addtrain_labels = np.array([0] * 24 + [1] * 22)
pitt_addtest_labels = np.array([0] * 5 + [1] * 5)


conn_est = ConnectivityMeasure(kind='tangent')
pitt_addfeatures_train = conn_est.fit_transform(pitt_addfeatures_train)
pitt_addfeatures_train = sym_matrix_to_vec(pitt_addfeatures_train)

pitt_addfeatures_test = conn_est.fit_transform(pitt_addfeatures_test)
pitt_addfeatures_test = sym_matrix_to_vec(pitt_addfeatures_test)


'''
加载sbl数据
'''
sbl_train_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\sbl\\sbl_train_features.npy')
sbl_train_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\sbl\\sbl_train_labels.npy')
sbl_train_features = np.squeeze(sbl_train_features)
# print("sbl_train_features.shape:", sbl_train_features.shape)
# print("sbl_train_labels.shape", sbl_train_labels.shape)

sbl_test_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\sbl\\sbl_test_features.npy')
sbl_test_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\sbl\\sbl_test_labels.npy')
sbl_test_features = np.squeeze(sbl_test_features)
# print("sbl_test_features.shape:", sbl_test_features.shape)
# print("sbl_test_labels.shape", sbl_test_labels.shape)

#sbl构建皮尔逊矩阵
sbl_addfeatures_train = []
sbl_addfeatures_test = []

for i in range(24):
    min_train = sbl_train_features[i * 196:(i + 1) * 196]
    sbl_addfeatures_train.append(min_train)

for i in range(5):
    min_test = sbl_test_features[i * 196:(i + 1) * 196]
    sbl_addfeatures_test.append(min_test)

sbl_addfeatures_train = np.squeeze(np.array(sbl_addfeatures_train))
sbl_addfeatures_test = np.squeeze(np.array(sbl_addfeatures_test))
sbl_addtrain_labels = np.array([0] * 12 + [1] * 12)
sbl_addtest_labels = np.array([0] * 2 + [1] * 3)


conn_est = ConnectivityMeasure(kind='tangent')
sbl_addfeatures_train = conn_est.fit_transform(sbl_addfeatures_train)
sbl_addfeatures_train = sym_matrix_to_vec(sbl_addfeatures_train)

sbl_addfeatures_test = conn_est.fit_transform(sbl_addfeatures_test)
sbl_addfeatures_test = sym_matrix_to_vec(sbl_addfeatures_test)

'''
加载sdsu数据
'''
sdsu_train_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\sdsu\\sdsu_train_features.npy')
sdsu_train_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\sdsu\\sdsu_train_labels.npy')
sdsu_train_features = np.squeeze(sdsu_train_features)
# print("sdsu_train_features.shape:", sdsu_train_features.shape)
# print("sdsu_train_labels.shape", sdsu_train_labels.shape)

sdsu_test_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\sdsu\\sdsu_test_features.npy')
sdsu_test_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\sdsu\\sdsu_test_labels.npy')
sdsu_test_features = np.squeeze(sdsu_test_features)
# print("sdsu_test_features.shape:", sdsu_test_features.shape)
# print("sdsu_test_labels.shape", sdsu_test_labels.shape)

#sdsu构建皮尔逊矩阵
sdsu_addfeatures_train = []
sdsu_addfeatures_test = []

for i in range(30):
    min_train = sdsu_train_features[i * 176:(i + 1) * 176]
    sdsu_addfeatures_train.append(min_train)

for i in range(6):
    min_test = sdsu_test_features[i * 176:(i + 1) * 176]
    sdsu_addfeatures_test.append(min_test)

sdsu_addfeatures_train = np.squeeze(np.array(sdsu_addfeatures_train))
sdsu_addfeatures_test = np.squeeze(np.array(sdsu_addfeatures_test))
sdsu_addtrain_labels = np.array([0] * 12 + [1] * 18)
sdsu_addtest_labels = np.array([0] * 2 + [1] * 4)


conn_est = ConnectivityMeasure(kind='tangent')
sdsu_addfeatures_train = conn_est.fit_transform(sdsu_addfeatures_train)
sdsu_addfeatures_train = sym_matrix_to_vec(sdsu_addfeatures_train)

sdsu_addfeatures_test = conn_est.fit_transform(sdsu_addfeatures_test)
sdsu_addfeatures_test = sym_matrix_to_vec(sdsu_addfeatures_test)


'''
加载um数据
'''
um_train_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\um\\um_train_features.npy')
um_train_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\um\\um_train_labels.npy')
um_train_features = np.squeeze(um_train_features)
# print("um_train_features.shape:", um_train_features.shape)
# print("um_train_labels.shape", um_train_labels.shape)

um_test_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\um\\um_test_features.npy')
um_test_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\um\\um_test_labels.npy')
um_test_features = np.squeeze(um_test_features)
# print("um_test_features.shape:", um_test_features.shape)
# print("um_test_labels.shape", um_test_labels.shape)

#um构建皮尔逊矩阵
um_addfeatures_train = []
um_addfeatures_test = []

for i in range(86):
    min_train = um_train_features[i * 296:(i + 1) * 296]
    um_addfeatures_train.append(min_train)

for i in range(20):
    min_test = um_test_features[i * 296:(i + 1) * 296]
    um_addfeatures_test.append(min_test)

um_addfeatures_train = np.squeeze(np.array(um_addfeatures_train))
um_addfeatures_test = np.squeeze(np.array(um_addfeatures_test))
um_addtrain_labels = np.array([0] * 43 + [1] * 43)
um_addtest_labels = np.array([0] * 10 + [1] * 10)


conn_est = ConnectivityMeasure(kind='tangent')
um_addfeatures_train = conn_est.fit_transform(um_addfeatures_train)
um_addfeatures_train = sym_matrix_to_vec(um_addfeatures_train)

um_addfeatures_test = conn_est.fit_transform(um_addfeatures_test)
um_addfeatures_test = sym_matrix_to_vec(um_addfeatures_test)


'''
加载usm数据
'''
usm_train_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\usm\\usm_train_features.npy')
usm_train_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\usm\\usm_train_labels.npy')
usm_train_features = np.squeeze(usm_train_features)
# print("usm_train_features.shape:", usm_train_features.shape)
# print("usm_train_labels.shape", usm_train_labels.shape)

usm_test_features = np.load('C:\\Users\\atr\\Desktop\\save\\single\\usm\\usm_test_features.npy')
usm_test_labels = np.load('C:\\Users\\atr\\Desktop\\save\\single\\usm\\usm_test_labels.npy')
usm_test_features = np.squeeze(usm_test_features)
# print("usm_test_features.shape:", usm_test_features.shape)
# print("usm_test_labels.shape", usm_test_labels.shape)

#usm构建皮尔逊矩阵
usm_addfeatures_train = []
usm_addfeatures_test = []

for i in range(57):
    min_train = usm_train_features[i * 236:(i + 1) * 236]
    usm_addfeatures_train.append(min_train)

for i in range(14):
    min_test = usm_test_features[i * 236:(i + 1) * 236]
    usm_addfeatures_test.append(min_test)

usm_addfeatures_train = np.squeeze(np.array(usm_addfeatures_train))
usm_addfeatures_test = np.squeeze(np.array(usm_addfeatures_test))
usm_addtrain_labels = np.array([0] * 37 + [1] * 20)
usm_addtest_labels = np.array([0] * 9 + [1] * 5)


conn_est = ConnectivityMeasure(kind='tangent')
usm_addfeatures_train = conn_est.fit_transform(usm_addfeatures_train)
usm_addfeatures_train = sym_matrix_to_vec(usm_addfeatures_train)

usm_addfeatures_test = conn_est.fit_transform(usm_addfeatures_test)
usm_addfeatures_test = sym_matrix_to_vec(usm_addfeatures_test)


'''
构建多站点数据集
'''

train_data = np.vstack((caltech_addfeatures_train,
                        leuven_addfeatures_train,
                        nyu_addfeatures_train,
                        ohsu_addfeatures_train,
                        olin_addfeatures_train,
                        pitt_addfeatures_train,
                        sbl_addfeatures_train,
                        sdsu_addfeatures_train,
                        um_addfeatures_train,
                        usm_addfeatures_train,
                        ))
train_label = np.hstack((caltech_addtrain_labels,
                         leuven_addtrain_labels,
                         nyu_addtrain_labels,
                         ohsu_addtrain_labels,
                         olin_addtrain_labels,
                         pitt_addtrain_labels,
                         sbl_addtrain_labels,
                         sdsu_addtrain_labels,
                         um_addtrain_labels,
                         usm_addtrain_labels,
                         ))

test_data = np.vstack((caltech_addfeatures_test,
                       leuven_addfeatures_test,
                       nyu_addfeatures_test,
                       ohsu_addfeatures_test,
                       olin_addfeatures_test,
                       pitt_addfeatures_test,
                       sbl_addfeatures_test,
                       sdsu_addfeatures_test,
                       um_addfeatures_test,
                       usm_addfeatures_test,
                       ))
test_label = np.hstack((caltech_addtest_labels,
                        leuven_addtest_labels,
                        nyu_addtest_labels,
                        ohsu_addtest_labels,
                        olin_addtest_labels,
                        pitt_addtest_labels,
                        sbl_addtest_labels,
                        sdsu_addtest_labels,
                        um_addtest_labels,
                        usm_addtest_labels,
                        ))


print("train_data:", train_data.shape)
print("train_label:", train_label.shape)
print("test_data:", test_data.shape)
print("test_label:", test_label.shape)



















