import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from scipy.special import expit


class Logistic_regression():

    def __init__(self, x_train, y_train, x_test, y_test, lr=0.001, epochs=5):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.learning_rate = lr
        self.epochs = epochs

    def sigmoid(self, X):
        return expit(X)

    def gradAscent(self, X, y, return_weights_only=True):
        """
        Params:
            X:训练样本
            y:训练样本标签
            weights:求得的回归系数数组(最优参数θ)
        """
        xMatrix = np.mat(X)
        yMatrix = np.mat(y).T
        rows, cols = np.shape(X)
        weights = np.ones((cols, 1))
        weights_array = np.array([])
        for epoch in range(self.epochs):
            h = xMatrix * weights
            y_pred = self.sigmoid(h)
            error = yMatrix - y_pred
            w_grad = xMatrix.T.dot(error)
            weights = weights + self.learning_rate * w_grad
            weights_array = np.append(weights_array, weights)
        weights_array = weights_array.reshape(self.epochs, cols)
        if return_weights_only:
            return weights
        else:
            return weights, weights_array

    def classifyVector(self, sample, weights, threshold=0.5):
        """
        根据训练好的权重向量，对输入样本sample根据阈值，进行分类
        """
        y_pred = self.sigmoid(sum(sample * weights))
        if y_pred > threshold:
            return 1.0
        else:

            return 0.0

    def accTest(self):
        """
        使用训练集训练，之后使用得到的权重在测试集上测试，并返回准确率

        """

        x_train = np.array(self.x_train)
        y_train = np.array(self.y_train)
        x_test = np.array(self.x_test)
        y_test = np.array(self.y_test)

        trainWeights = self.gradAscent(x_train, y_train, return_weights_only=True)
        # 定义变量，用于保存分对样本的个数
        trueCount = 0
        # 定义变量，用于保存测试样本的总个数
        testCount = 0

        for labels in y_test:

            currLine = x_test[testCount]
            testCount += 1
            test_pred = self.classifyVector(currLine, trainWeights)
            true_label = labels

            if test_pred == true_label:
                trueCount += 1

        # testCount 为float类型
        accuracy = float(trueCount / testCount) * 100
        print(f'The accuracy of using your own Logistic Regression is {round(accuracy, 4)}%')

        return accuracy


if __name__ == '__main__':
    # 导入数据集iris  
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    iris = pandas.read_csv(url, names=names)

    # 数据预处理
    X = iris[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
    # X = X[0:100]
    y = iris['class']
    # y = y[0:100]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=101)

    # 使用 sklearn 库函数计算
    model = LogisticRegression()
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    print('The accuracy of the Logistic Regression is: {0}'.format(metrics.accuracy_score(prediction, test_y)))

    # 使用 bp 的 logistics 计算(SGD)
    model = Logistic_regression(x_train=train_X, y_train=train_y, x_test=test_X, y_test=test_y)
    model.epochs = 500
    model.accTest()
