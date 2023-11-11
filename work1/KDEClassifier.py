import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator, ClassifierMixin

class KDEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = []
        for label in self.classes_:
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(X[y == label])
            self.models_.append(kde)
        return self
        
    def predict(self, X):
        logprobs = np.array([model.score_samples(X) for model in self.models_]).T
        return self.classes_[np.argmax(logprobs, axis=1)]

# # 加载数据
# iris = load_iris()
# X, y = iris.data, iris.target

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 训练模型
# clf = KDEClassifier(bandwidth=1.0)
# clf.fit(X_train, y_train)

# # 预测
# y_pred = clf.predict(X_test)

# # 输出准确率
# accuracy = np.mean(y_pred == y_test)
# print(f'Accuracy: {accuracy * 100:.2f}%')
