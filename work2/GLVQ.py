import numpy as np

class GLVQ:
    def __init__(self, n_prototypes, learning_rate=0.01, n_iterations=100):
        self.n_prototypes = n_prototypes
        self.learning_rate = learning_rate
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
                closest_same, closest_diff = self._find_closest_prototypes(x, label)

                # 计算距离
                dist_closest_same = np.linalg.norm(x - closest_same[0])
                dist_closest_diff = np.linalg.norm(x - closest_diff[0])

                # 计算梯度并更新原型
                factor = (dist_closest_same - dist_closest_diff) / (dist_closest_same + dist_closest_diff)
                closest_same[0] -= self.learning_rate * factor * (x - closest_same[0])
                closest_diff[0] += self.learning_rate * factor * (x - closest_diff[0])

    def _find_closest_prototypes(self, x, label):
        closest_same = None
        closest_diff = None
        min_dist_same = float('inf')
        min_dist_diff = float('inf')

        for prototype in self.prototypes:
            dist = np.linalg.norm(x - prototype[0])
            if prototype[1] == label and dist < min_dist_same:
                closest_same = prototype
                min_dist_same = dist
            elif prototype[1] != label and dist < min_dist_diff:
                closest_diff = prototype
                min_dist_diff = dist

        return closest_same, closest_diff

    def predict(self, X):
        predictions = []
        for x in X:
            closest_prototype = min(self.prototypes, key=lambda p: np.linalg.norm(x - p[0]))
            predictions.append(closest_prototype[1])
        return np.array(predictions)

# 示例使用
# X, y = 加载您的数据
# model = GLVQ(n_prototypes=2)
# model.fit(X, y)
# predictions = model.predict(X
