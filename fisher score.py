'''
Fisher score 是特征选择的一种算法
公式如下所示：

F(i) = ((p_average - average) ** 2 + (n_average - average) ** 2) /
                   (np.sum(p_var) / len(p) + np.sum(n_var) / len(n))

该公式是对每个列向量求一个fisher score, 然后选择出Fisher score中前n个最大的得分
所对应的列向量作为新的数据特征
'''



import numpy as np

def fisherscore(data, labels, num):

    high = len(data)  # 向量个数
    weight = len(data[0])  # 向量长度
    P_num = np.sum(labels == 0)  # 正样本
    N_num = np.sum(labels == 1)  # 负样本

    # 计算Fisher score

    fisherscore = []
    for i in range(weight):
        p = []
        n = []
        p_var = []
        n_var = []
        for j in range(high):
            if labels[j] == 0:
                p.append(data[j, i])
            if labels[j] == 1:
                n.append(data[j, i])

        p_average = np.sum(p) / len(p)
        n_average = np.sum(n) / len(n)
        average = (np.sum(p) + np.sum(n)) / (len(p) + len(n))

        for j in range(high):
            if labels[j] == 0:
                p_var.append((data[j, i] - p_average) ** 2)
            if labels[j] == 1:
                n_var.append((data[j, i] - n_average) ** 2)

        score = ((p_average - average) ** 2 + (n_average - average) ** 2) / (
                    np.sum(p_var) / len(p) + np.sum(n_var) / len(n))

        fisherscore.append(score)

    index = np.argsort(-np.array(fisherscore))  # 返回索引
    new_data = []
    for i in range(num):
        new_data.append(data[:, index[i]])

    new_data = np.array(new_data)
    new_data = new_data.transpose(1, 0)

    return new_data

data = np.array([ [1, 2, 3, 4, 5, 0], [6, 7, 8, 9, 0, 6], [3, 5, 6, 8, 1, 4],
                [4, 5, 7, 9, 3, 5], [5, 6, 8, 0, 2, 1]])
labels = np.array([0]*3+[1]*2)
num = 3
newdata = fisherscore(data, labels, num)


print(newdata)
print(newdata.shape)