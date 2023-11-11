# @file     MQDF.py
#
# @date     2023-04
#
# @brief    Python code for INT301 Lab. Discriminant Functions & Non-parametric Classifiers
#           This code will implement the MQDF algorithm for iris.data classification
#           without using any third-party algorithm library.

# ----------------------------------------------------------------------------------------------------------- #
###############################################################################################################
#                             You need to fill the missing part of the code                                   #
#                        detailed instructions have been given in each function                              #
###############################################################################################################
# ----------------------------------------------------------------------------------------------------------- #


import math
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import timeit
from math import *
from Data_process import Data_process
import logging
import os
from pathlib import Path

###############################################################################################################
#                                                  log                                                        #
###############################################################################################################

def create_log():
    log_dir = Path('./log/')
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("MQDF")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s_log.txt' % (log_dir, "MQDF"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def log_string(logger, str):
    logger.info(str)
    print(str)

###############################################################################################################
#                                                  draw                                                       #
###############################################################################################################

def draw(x, y, xlabel, ylabel, fileName):
    x = np.asarray(x)
    x = x.reshape(x.shape[0], -1)
    y = np.asarray(y)
    y = y.reshape(x.shape[0], -1)
    plt.figure(facecolor='w',edgecolor='w')
    plt.plot(x, y, linestyle = '-', linewidth = '1.5')
    plt.xlabel(xlabel, fontsize='x-large')
    plt.ylabel(ylabel, fontsize='x-large')
    plt.grid()
    plt.savefig('./log/'+fileName+'.png', dpi=600, format='png')
    temp_array = np.append(x, y, axis=1)
    np.savetxt('./log/'+fileName+'.txt', temp_array, fmt='%2f', delimiter =",")

###############################################################################################################
#                                        Self-defined functions                                               #
###############################################################################################################

def pz_predict(x_len, np_array):  # To find the predicted class of the test set
    x_pred = []
    for i in range(x_len):
        max = np.max(np_array[:, i])  # Get the maximum probability in the ith column (ith data of x)
        if max == np_array[0][i]:  # If 'max' is equal to the ith value in setosa array
            pred = 'Iris-setosa'
        elif max == np_array[1][i]:  # If 'max' is equal to the ith value in versicolor array
            pred = 'Iris-versicolor'
        else:  # If 'max' is equal to the ith value in virginica array
            pred = 'Iris-virginica'
        x_pred.append(pred)  # Store the predicted class in the order of the test datasets
    return x_pred


def pz_accuracy(pred_class, class_x):  # To obtain the accuracy of the predicted result
    acc = 0  # Initialize the accuracy
    for ind, pred in enumerate(pred_class):
        if pred == class_x[ind]:  # Compare the predicted classes with the actual classes of the test set
            acc += 1  # Increase the accuracy parameter if it is correct
        else:
            pass  # If not correct, pass
    return (acc / len(pred_class) * 100)


###############################################################################################################
#                                   Self-defined functions for MQDF                                           #
###############################################################################################################

def QDF_model(train_data, class_num):  # Modified function from QDF to obtain the required parameters for MQDF
    mean = []  # Initialize a list to store the mean
    cov_matrix = []  # Initialize a list to store covariance matrices
    for i in range(class_num):  # For all classes
        ###############################################################################################################
        #                                   YOU NEED FILL FOLLOWING CODES:
        train_data[i] = np.matrix(train_data[i], dtype=np.float64)
        mean.append(train_data[i].mean(0).T)
        cov_matrix.append(np.matrix(np.cov(train_data[i].T)))
        # Tip: use np.cov, consider a row as one feature. Pay attention to the data that may should be transposed
        ###############################################################################################################
    return mean, cov_matrix

def MQDF1_model(cov, d, class_num, h):  # Function to obtain the trained parameters of MQDF1
    eigenvalue = []  # Initialize a list to store eigenvalues
    eigenvector = []  # Initialize a list to store eigenvectors
    delta = [0] * class_num  # Each delta value of classes will be stored here
    # h2I = np.eye(d) * (h**2)

    for i in range(class_num):  # For all classes
        ###############################################################################################################
        #                                   YOU NEED FILL FOLLOWING CODES:
        covMat = cov[i] # Get the covariance matrix of ith class
        eigvals, eigvecs = np.linalg.eig(covMat) # Obtain eigenvalues and eigenvectors from the cov. matrix. Tip: usenp.linalg.eig()
        eigenvector.append(eigvecs)  # append eigvecs to eigenvector
        eigenvalue.append(eigvals + h**2)  # add h^2 to eigvals, append eigvals to eigenvalue
        delta[i] = np.mean(eigvals) # Compute delta as the mean of minor values.
        ###############################################################################################################
    return eigenvalue, eigenvector, delta

def MQDF1_model_h(cov, d, class_num):  # Function to obtain the trained parameters of MQDF1
    eigenvalue = []  # Initialize a list to store eigenvalues
    eigenvector = []  # Initialize a list to store eigenvectors
    delta = [0] * class_num  # Each delta value of classes will be stored here
    # h2I = np.eye(d) * (h**2)

    for i in range(class_num):  # For all classes
        ###############################################################################################################
        #                                   YOU NEED FILL FOLLOWING CODES:
        covMat = cov[i] # Get the covariance matrix of ith class
        eigvals, eigvecs = np.linalg.eig(covMat) # Obtain eigenvalues and eigenvectors from the cov. matrix. Tip: usenp.linalg.eig()
        eigenvector.append(eigvecs)  # append eigvecs to eigenvector
        h2 = np.mean(eigvals)
        eigenvalue.append(eigvals + h2)  # add h^2 to eigvals, append eigvals to eigenvalue
        delta[i] = np.mean(eigvals) # Compute delta as the mean of minor values.
        ###############################################################################################################
    return eigenvalue, eigenvector, delta, h2

def predict_MQDF1(d, np_x, class_num, mean, eigenvalue, eigenvector,
                  delta):  # Function to perform classification based on the MQDF1 trained parameters
    # assert (k < d and k > 0)  # Assertion error when k greater or equal to d and negative k
    pred_label = []  # Initialize a list to store the predicted classes
    for sample in np_x:  # For the number of test samples
        test_x = np.matrix(sample, np.float64).T  # Convert a sample data to a matrix
        max_g = -float('inf')  # The initial value of max_g2 is set to the negative infinity
        for i in range(class_num):  # For all classes
            dis = np.linalg.norm(test_x.reshape((d,)) - mean[i].reshape(
                (d,))) ** 2  # Compute the distance between the sample data and the mean, and then square it
            # Second term of the residual of subspace projection
            euc_dis = [0] * int(d)  # Initialization
            ma_dis = [0] * int(d)  # Initialization
            for j in range(int(d)):  # For the range of d
                euc_dis[j] = ((eigenvector[i][:, j].T * (test_x - mean[i]))[0, 0]) ** 2
                ma_dis[j] = (((test_x - mean[i]).T * eigenvector[i][:, j])[0, 0]) ** 2

            g = 0  # Initialize the MQDF1
            for j in range(int(d)):  # For the range of d
                # Firstly, compute the terms including j and add them to g
                g += (euc_dis[j] * 1.0 / eigenvalue[i][j]) + math.log(eigenvalue[i][j])

            # Secondly, compute the terms only including i and add them to g
            g += ((dis - sum(ma_dis)) / delta[i])
            g = -g  # Convert the g2 values to minus to find the maximum value

            if g > max_g:  # If the current g > previous max g
                max_g = g  # Replace the g value
                prediction = i  # Store the class id of current g
            elif g == max_g:
                print(i, "==", prediction)  # Error if two g values are equal
            else:
                pass  # Ignore current g if it's smaller than max_g
        pred_label.append(prediction)  # After for loop, append the current max g class id
    return pred_label

def MQDF2_model(cov, d, k, class_num):  # Function to obtain the trained parameters of MQDF2
    eigenvalue = []  # Initialize a list to store eigenvalues
    eigenvector = []  # Initialize a list to store eigenvectors
    delta = [0] * class_num  # Each delta value of classes will be stored here
    for i in range(class_num):  # For all classes
        ###############################################################################################################
        #                                   YOU NEED FILL FOLLOWING CODES:
        covMat = cov[i] # Get the covariance matrix of ith class
        eigvals, eigvecs = np.linalg.eig(covMat) # Obtain eigenvalues and eigenvectors from the cov. matrix. Tip: usenp.linalg.eig()
        # Disordering the eigenvalues
        id = eigvals.argsort()# Get the ascending order of the eigenvalue indexes. Tip: use array.argsort()
        id = id[::-1] # Convert id to the descending order of the eigenvalues. Tip: use slicing as [::-1]
        eigvals = eigvals[id] # Descending the eigenvalues
        eigvecs = eigvecs[:,id] # Descending the eigenvectors
        eigenvector.append(eigvecs[:,:k])  # Store the eigenvectors from j=1 to k
        eigenvalue.append(eigvals[:k])  # Store the eigenvalues from j=1 to k
        delta[i] = np.sum(eigvals[k:]) / (d-k) # Compute delta as the mean of minor values. Tip: sum(eigvals[int(k):]) / (d - k)
        # ###############################################################################################################
    return eigenvalue, eigenvector, delta


def predict_MQDF2(d, np_x, class_num, k, mean, eigenvalue, eigenvector, delta): # Function to perform classification based on the MQDF2 trained parameters
    # assert (k < d and k > 0)  # Assertion error when k greater or equal to d and negative k
    pred_label = [] # Initialize a list to store the predicted classes
    for sample in np_x: # For the number of test samples
        test_x = np.matrix(sample, np.float64).T # Convert a sample data to a matrix
        max_g2 = -float('inf') # The initial value of max_g2 is set to the negative infinity
        for i in range(class_num): # For all classes
            dis = np.linalg.norm(test_x.reshape((d,)) - mean[i].reshape((d,))) ** 2 # Compute the distance between the sample data and the mean, and then square it
            # Second term of the residual of subspace projection
            euc_dis = [0] * int(k) # Initialization
            ma_dis = [0] * int(k)  # Initialization
            for j in range(int(k)): # For the range of k
                euc_dis[j] = ((eigenvector[i][:, j].T * (test_x - mean[i]))[0, 0]) ** 2
                ma_dis[j] = (((test_x - mean[i]).T * eigenvector[i][:, j])[0, 0]) ** 2

            g2 = 0  # Initialize the MQDF2
            for j in range(int(k)): # For the range of k
                # Firstly, compute the terms including j and add them to g2
                g2 += (euc_dis[j] * 1.0 / eigenvalue[i][j]) + math.log(eigenvalue[i][j])

            # Secondly, compute the terms only including i and add them to g2
            g2 += ((dis - sum(ma_dis)) / delta[i]) + ((d - int(k)) * math.log(delta[i]))
            g2 = -g2 # Convert the g2 values to minus to find the maximum value

            if g2 > max_g2: # If the current g2 > previous max g2
                max_g2 = g2 # Replace the g2 value
                prediction = i # Store the class id of current g2
            elif g2 == max_g2:
                print(i, "==", prediction) # Error if two g2 values are equal
            else:
                pass # Ignore current g2 if it's smaller than max_g2
        pred_label.append(prediction) # After for loop, append the current max g2 class id
    return pred_label

def mqdf_accuracy(predic, class_x): # Function to calculate the prediction accuracy of MQDF2
    conv_pred = []
    for pred in predic: # For the predicted id numbers of test dataset
        if pred == 0: # If the predicted value is '0'
            conv_pred.append('Iris-setosa') # It is setosa
        elif pred == 1: # If it is '1'
            conv_pred.append('Iris-versicolor') # It is versicolor
        elif pred == 2: # If it is '2'
            conv_pred.append('Iris-virginica') # It is virginica
        else: # Out of range, there is an error
            print('Wrong prediction')  # Error when index out of the range

    accuracy = pz_accuracy(conv_pred, class_x) # Accuracy can be calculated using 'pz_accuracy()'
    return accuracy


###############################################################################################################
#                                              Main Part                                                      #
###############################################################################################################


if __name__ == '__main__':
    logger = create_log()
    log_string(logger, 'starting...')
    import os
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    iris = Data_process() # Define class
    irist_data = iris.load_data() # Load the iris dataset
    div_data = iris.numeric_n_name(irist_data) # Separate numeric dataset and class names
    init_data = iris.shuffle() # Shuffle the dataset
    five_data = iris.separate_data() # Divide the dataset for 5-fold cross validation
    # Predefined parameters
    classNum = 3  # Three classes
    d = len(div_data[0][0])  # Feature numbers = 4

    k_list = []
    h_list = []
    t1_list = []
    t2_list = []
    mqdf1_acc_list = []
    mqdf2_acc_list = []

    log_string(logger, 'loaded training set')

    h = 10
    mqdf_sum_avg_acc = 0
    cnt = 1
    start = timeit.default_timer()  # To measure the running time of MQDF1, start timer
    for index in range(len(five_data)): # 5-fold cross validation
        total_subset = iris.combine_train(index, five_data)  # Index denotes the array for testing
        sep_dataset = iris.separate_class(total_subset[0])  # Return separated train datasets by three classes
        sep_data = [sep_dataset[0], sep_dataset[1], sep_dataset[2]] # Nested list of the three datasets
        # Only extract the numeric data from the datasets
        np_se = np.array(iris.numeric_n_name(sep_data[0])[0])
        np_ver = np.array(iris.numeric_n_name(sep_data[1])[0])
        np_vir = np.array(iris.numeric_n_name(sep_data[2])[0])
        # Prepare the train dataset by converting the numbers in 'str' to 'float
        train = [np_se.astype(float), np_ver.astype(float), np_vir.astype(float)]
        mean, cov = QDF_model(train, classNum) # Obtain the mean and covariance matrices
        eigval, eigvec, delta = MQDF1_model(cov, d, classNum, h= h*.01) # Obtain the eigenvalues, eigenvectors and delta
        # mean, eigenvalues, eigenvectors, k and delta will be the trained parameters of MQDF2 for prediction
        # print(f'Training process of the MQDF1 {index} model finished.')
        # log_string(logger, 'Training process of the MQDF1 %d model finished' %(index))

        # Prepare the test dataset
        x = np.array(iris.numeric_n_name(total_subset[1])[0])  # numeric data of test set
        np_x = x.astype(float)
        x_len = len(np_x)

        class_x = iris.numeric_n_name(total_subset[1])[1]  # Real class names of each test set
        predic = predict_MQDF1(d, np_x, classNum, mean, eigval, eigvec, delta) # Input the trained parameters to predict

        MQDF_accuracy = mqdf_accuracy(predic, class_x) # Based on the prediction result, compute the classification accuracy
        mqdf_sum_avg_acc += MQDF_accuracy
        # print(cnt, 'th accuracy:', MQDF_accuracy)
        cnt += 1
    stop = timeit.default_timer() # Stop timer
    MQDF_avg_acc  = mqdf_sum_avg_acc / len(five_data) # Calculate the average accuracy of 5-fold cross validation
    # print('Averay accuracy of 5-fold cross-validation when h =', h*.01, ':', MQDF_avg_acc)
    if (h % 10) == 0:
        log_string(logger, 'Averay accuracy of 5-fold cross-validation when h = %.2f is %.2f, running time is %.3f' %(h*.01, MQDF_avg_acc, (stop-start)))
    
    h_list.append(h*.01)
    t1_list.append(stop-start)
    mqdf1_acc_list.append(MQDF_avg_acc)
