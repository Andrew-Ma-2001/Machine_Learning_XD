import numpy as np
import pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


# 导入数据集iris  
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = pandas.read_csv(url, names=names)
X = iris[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]

y = iris['class']

encoder = LabelEncoder()
y = encoder.fit_transform(y)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=101)


# 假设每个特征服从高斯分布，选择高斯朴素贝叶斯来进行分类计算。
clf = GaussianNB(var_smoothing=1e-8)
clf.fit(train_X, train_y)
prediction = clf.predict(test_X)
print('The accuracy of the GaussianNB is: {0}'.format(metrics.accuracy_score(prediction, test_y)))
