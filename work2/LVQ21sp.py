import numpy as np

class LVQ21sp:
    def __init__(self, n_prototypes, learning_rate=0.01, window=0.3, n_iterations=100):
        self.n_prototypes = n_prototypes
        self.learning_rate = learning_rate
        self.window = window
        self.n_iterations = n_iterations
        self.prototypes = []

    def fit(self, X, y):
        unique_classes = np.unique(y)
        self.prototypes = []

        # 初始化原型
        for c in unique_classes:
            class_indices = np.where(y == c)[0]
            prototype_indices = np.random.choice(class_indices, self.n_prototypes, replace=False)
            self.prototypes.extend([[X[idx], c] for idx in prototype_indices])

        # 训练过程
        for _ in range(self.n_iterations):
            for x, label in zip(X, y):
                self._process_sample(x, label)

    def _process_sample(self, x, label):
        # 计算所有原型到当前样本的距离
        distances = np.array([np.linalg.norm(x - p[0]) for p in self.prototypes])
        prototype_labels = np.array([p[1] for p in self.prototypes])

        # 找到同类和异类最近的原型
        same_class = prototype_labels == label
        diff_class = ~same_class
        closest_same = np.argmin(np.where(same_class, distances, np.inf))
        closest_diff = np.argmin(np.where(diff_class, distances, np.inf))

        # 计算距离
        dist_closest_same = distances[closest_same]
        dist_closest_diff = distances[closest_diff]

        # 检查是否在窗口内
        if self._in_window(dist_closest_same, dist_closest_diff):
            # 更新原型
            self.prototypes[closest_same][0] += self.learning_rate * (x - self.prototypes[closest_same][0])
            self.prototypes[closest_diff][0] -= self.learning_rate * (x - self.prototypes[closest_diff][0])

    def _in_window(self, dist_same, dist_diff):
        s = (1 - self.window) / (1 + self.window)
        return s * dist_diff <= dist_same <= dist_diff / s

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.array([np.linalg.norm(x - p[0]) for p in self.prototypes])
            closest_prototype = np.argmin(distances)
            predictions.append(self.prototypes[closest_prototype][1])
        return np.array(predictions)

# 示例使用
# X, y = 加载您的数据
# model = LVQ21(n_prototypes=2)
# model.fit(X, y)
# predictions = model.predict(X)
