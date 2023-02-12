'''

* Logistic Regression
* sourse dataset : https://www.scilab.org/machine-learning-logistic-regression-tutorial

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#read the data file
df = pd.read_csv('data_classification.csv')

#split to train and test data
train, test = train_test_split(df, test_size = 0.3, random_state=40)
features_train = train.drop(columns = df.columns[-1])
labels_train = train[df.columns[-1]]
features_test = test.drop(columns = df.columns[-1])
labels_test = test[df.columns[-1]]

#add the bias
features_train['bias'] = pd.DataFrame(pd.Series(1,index = features_train.index))
features_test['bias'] = pd.DataFrame(pd.Series(1,index = features_test.index))

#convert to matrix
features_train = np.array(features_train)
labels_train = np.array([labels_train])#1x10
features_test = np.array(features_test)
labels_test = np.array([labels_test])

weights = np.array([[1.], [1.], [1.]])

'''matrixs push into functions
features: 100x3 (replace label with bias)
weight: 3x1
label: 100x1
'''

def  sigmoid(z):
    '''aglorthm
    pred = weights * feature
    z    = w1*x1 + w2*x2 + 1*1(bias)
    '''
    return 1.0/(1 + np.exp(-z))

def filter(pred):
    for item in pred:
        if item[-1] > 0.9999:
            item[-1] = 1
        else:
            item[-1] = 0
    return pred

def predict(features, weights):
    pred = np.dot(features, weights)
    return sigmoid(pred)

#cross entropy
def CE(features, weights, labels):
    prediction = predict(features, weights)
    n = len(labels)
    cost_class1 = -labels * np.log(prediction)
    cost_class2 = -(1-labels) * np.log((1 - prediction))
    cost= cost_class1 + cost_class2
    return cost.sum()/n

def update_weights(features, weights, labels, learning_rate):
    n = len(labels)
    prediction = predict(features, weights)
    gradient = np.dot(features.T, prediction -  labels)
    gradient = gradient/n
    gradient = gradient*learning_rate
    weights -= gradient
    return weights

def train(features, weights, labels, learning_rate, iter):
    Cost_history = [0]
    for i in range(iter):
        weights = update_weights(features, weights, labels, learning_rate)
        Cost = CE(features, weights, labels)
        Cost_history.append(Cost)
        if (Cost_history[-1] - Cost_history[-2] < 0.00001):
            break
    return weights, Cost_history

def ratio(result, labels):
    count = 0
    for i in range(len(result)):
        if int(result[i][0]) == labels_test[i][0]:
            count += 1
    ratio = (count / len(result))*100
    return ratio
    
learning_rate = 0.0001
iter = 1000

best_weights, Cost_history = train(features_train, weights, labels_train.T, learning_rate, iter)
prediction = predict(features_test, best_weights)
result = filter(prediction)
print(ratio(result, labels_test.T))
