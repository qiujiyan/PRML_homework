"""
XJTLU INT304 Assignment

Multi-classification problem based on QDF algorithm implementation

Refer to Modified Quadratic Discriminant Functions and the Application 
to Chinese Character Recognition by Kimura, Fumitaka and Takashina, 
Kenji and Tsuruoka, Shinji and Miyake, Yasuji

ref : https://github.com/Regen2001/MQDF-IRIS-recognition/blob/main/MQDF_for_lab.py#L100

"""
import math
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin



class MQDFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='gaussian', delta=None):
        self.delta = delta
        self.kernel = kernel

        
    def modify_covariance(self,cov):
        a, b = np.linalg.eig(cov)
        a_modeified = [0 for i in range(len(a))]
        sum_a = sum(a)
        c = 0
        for i in range(len(a)):
            index_i = np.where(a==sorted(a,reverse=True)[i])[0][0]
            a_modeified[index_i] = a[index_i]
            c += a[index_i]
            cc = float(c)/sum_a
            if self.delta == None :
                delta = float(sum_a-c)/float(len(a) - index_i + 1)
            else:
                delta = self.delta
            print('cc--',cc)
            while 0.3 <= cc:
                print('a_modeified[index_i] --',a_modeified[index_i] )

                a_modeified[index_i] = delta
                index_i +=1
                if len(a) == index_i:
                    break

        cov_modefied = np.dot(np.dot(np.linalg.inv(b), np.asmatrix(np.diag(a))),b)

        return cov_modefied
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classes_num = len(self.classes_)
        self.features_ = X.shape[1]
        # self.mean, self.cov = QDF_model(X, self.classes_num) # Obtain the mean and covariance matrices
        self.mean = []
        self.cov = []
        self.prior_probability = []
        for label in self.classes_:
            X_i = X[y == label]
            self.mean.append(X_i.mean(0))
            modify_covmatrix = self.modify_covariance(np.matrix(np.cov(X_i.T)))
            self.cov.append(modify_covmatrix)
            self.prior_probability.append( 1.0/ float(self.classes_num)) # 先验概率值
            
    def discriminant_function(self, x, covMatrix, mean_i, p_i):
        #covMatrix = self.data.cov().as_matrix(columns=None)
       # print covMatrix
        cov_inverse = np.linalg.pinv(covMatrix)
        a = x-mean_i
        b = np.dot(a.T, cov_inverse)
        c = -np.dot(b, a)

        discriminant_function_value = c + 2 * math.log(p_i)
        #print  "result:", surface_fuction_value
        return discriminant_function_value
    
    def point_classification(self, x):
        class_value = {}

        for class_i in range(self.classes_num):
            covMatrix = self.cov[class_i]
            mean_i = self.mean[class_i]
            p_i = self.prior_probability[class_i]
            class_value[class_i] = self.discriminant_function(x, covMatrix, mean_i, p_i)

        result_x = max(class_value, key=lambda key: class_value[key])
        return self.classes_[result_x]
    
    def predict(self, X):
        predic = []
        for i in X:
            result_i = self.point_classification(i)
            predic.append(result_i)
        return np.array(predic)