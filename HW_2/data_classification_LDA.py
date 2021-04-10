import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=0)
Iris1 = df.values[0:50, 0:4]
Iris2 = df.values[50:100, 0:4]
Iris3 = df.values[100:150, 0:4]

# 定义类均值向量
m1 = np.mean(Iris1, axis=0)
m2 = np.mean(Iris2, axis=0)
m3 = np.mean(Iris3, axis=0)

train_num = 25

# 定义类内离散度矩阵
s1 = np.zeros((4, 4))
s2 = np.zeros((4, 4))
s3 = np.zeros((4, 4))

for i in range(0, train_num, 1):
    a = Iris1[i, :] - m1
    a = np.array([a])
    b = a.T
    s1 = s1 + np.dot(b, a)
for i in range(0, train_num, 1):
    c = Iris2[i, :] - m2
    c = np.array([c])
    d = c.T
    s2 = s2 + np.dot(d, c)
    # s2=s2+np.dot((Iris2[i,:]-m2).T,(Iris2[i,:]-m2))
for i in range(0, train_num, 1):
    a = Iris3[i, :] - m3
    a = np.array([a])
    b = a.T
    s3 = s3 + np.dot(b, a)

# 定义总类内离散矩阵
sw12 = s1 + s2;
sw13 = s1 + s3;
sw23 = s2 + s3;

# 定义投影方向
a = np.array([m1 - m2])
sw12 = np.array(sw12, dtype='float')
sw13 = np.array(sw13, dtype='float')
sw23 = np.array(sw23, dtype='float')
# 判别函数以及阈值T（即w0）
a = m1 - m2
a = np.array([a])
a = a.T
b = m1 - m3
b = np.array([b])
b = b.T
c = m2 - m3
c = np.array([c])
c = c.T
w12 = (np.dot(np.linalg.inv(sw12), a)).T
w13 = (np.dot(np.linalg.inv(sw13), b)).T
w23 = (np.dot(np.linalg.inv(sw23), c)).T
T12 = -0.5 * (np.dot(np.dot((m1 + m2), np.linalg.inv(sw12)), a))
T13 = -0.5 * (np.dot(np.dot((m1 + m3), np.linalg.inv(sw13)), b))
T23 = -0.5 * (np.dot(np.dot((m2 + m3), np.linalg.inv(sw23)), c))
# 计算正确率
kind1 = 0
kind2 = 0
kind3 = 0
newiris1 = []
newiris2 = []
newiris3 = []

for i in range(train_num, 49):
    x = Iris1[i, :]
    x = np.array([x])
    g12 = np.dot(w12, x.T) + T12
    g13 = np.dot(w13, x.T) + T13
    g23 = np.dot(w23, x.T) + T23
    if g12 > 0 and g13 > 0:
        newiris1.extend(x)
        kind1 = kind1 + 1
    elif g12 < 0 and g23 > 0:
        newiris2.extend(x)
    elif g13 < 0 and g23 < 0:
        newiris3.extend(x)

for i in range(train_num, 49):
    x = Iris2[i, :]
    x = np.array([x])
    g12 = np.dot(w12, x.T) + T12
    g13 = np.dot(w13, x.T) + T13
    g23 = np.dot(w23, x.T) + T23
    if g12 > 0 and g13 > 0:
        newiris1.extend(x)
    elif g12 < 0 and g23 > 0:

        newiris2.extend(x)
        kind2 = kind2 + 1
    elif g13 < 0 and g23 < 0:
        newiris3.extend(x)
for i in range(train_num, 49):
    x = Iris3[i, :]
    x = np.array([x])
    g12 = np.dot(w12, x.T) + T12
    g13 = np.dot(w13, x.T) + T13
    g23 = np.dot(w23, x.T) + T23
    if g12 > 0 and g13 > 0:
        newiris1.extend(x)
    elif g12 < 0 and g23 > 0:
        newiris2.extend(x)
    elif g13 < 0 and g23 < 0:
        newiris3.extend(x)
        kind3 = kind3 + 1

correct = (kind1 + kind2 + kind3) / ((50 - train_num) * 3)
correct_k1 = (kind1/(50 - train_num))
correct_k2 = (kind2/(50 - train_num))
correct_k3 = (kind3/(50 - train_num))
print('1,2 accuracy', correct_k1)
print('1,3 accuracy', correct_k2)
print('2,3 accuracy', correct_k3)
print('Average Accuracy', correct)




y1 = np.zeros(50)
y2 = np.zeros(50)
y3 = np.zeros(50)

plt.scatter(df.values[:50, 4], df.values[:50, 2], color='red', marker='o', label='setosa')
plt.scatter(df.values[50:100, 4], df.values[50: 100, 2], color='blue', marker='x', label='versicolor')
plt.scatter(df.values[100:150, 4], df.values[100: 150, 2], color='green', label='virginica')

plt.title("1,2,3")
plt.legend(loc='upper left')
plt.show()