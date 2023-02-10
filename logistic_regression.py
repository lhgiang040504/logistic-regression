import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("data_classification.csv", header=None)

train, test = train_test_split(df, test_size = 0.3, random_state=40)

features_train = train.drop(columns = df.columns[-1])
labels_train = train[df.columns[-1]]

features_test = test.drop(columns = df.columns[-1])
labels_test = test[df.columns[-1]]

features_train['bias'] = pd.DataFrame(pd.Series(1,index = features_train.index))
features_test['bias'] = pd.DataFrame(pd.Series(1,index = features_train.index))

labels_train = np.array(labels_train)
print(labels_train)

weights = np.array([[1],
[1],
[1]])

'''
x_true = []
y_true = []
x_false = []
y_false = []
#1x100

for item in df.values:
    if item[-1] == 1:
        x_true.append(item[0])
        y_true.append(item[1])
    else:
        x_false.append(item[0])
        y_false.append(item[1])


plt.plot(x_true, y_true, 'o', c='blue')
plt.plot(x_false, y_false, 'o', c='red')
plt.show()
'''

def  sigmoid(z):
    '''aglorthm
    pred = weights * feature
    z    = w1*x1 + w2*x2 + 1*1(bias)
    '''
    return 1.0/(1 + np.exp(-z))

def filter_data(pred):
    if pred >= 0.9999:
        return 1
    else:
        return 0
def predict(features, weights):
    pred = np.dot(features, weights)
    return sigmoid(pred)

def CE(features, weights, labels):
    '''matrics
    features: 100x3 (replace label with bias)
    weight: 1x3
    label: 100x1
    '''
    prediction = predict(features, weights)
    n = len(labels)
    cost_class1 = -labels * prediction
    cost_class2 = -(1-labels) * (1 - prediction)
    cost= cost_class1 + cost_class2
    return cost.sum()/n

def update_weights(features, weights, labels, learning_rate):
    '''matric
    features: 100x3 (replace label with bias)
    weight: 1x3
    label: 100x1
    learning_rate: float
    '''
    n = len(labels)
    prediction = predict(features, weights)
    gradient = np.dot(features, prediction -  labels)
    gradient = gradient/n
    gradient = gradient*learning_rate
    weights -= gradient
    return weights

def train(features, weights, labels, learning_rate, iter):
    Cost_history = []
    for i in range(iter):
        weights = update_weights(features, weights, labels, learning_rate)
        Cost = CE(features, weights, labels)
        Cost_history.append(Cost)
        if (Cost_history[-1] - Cost_history[-2] < 0.001):
            break
    return weights, Cost_history

learning_rate = 0.001
iter = 1000

#best_weights, Cost_history = train(features_train, weights, labels_train, learning_rate, iter)
prediction = predict(features_train, weights)
n = len(labels_train)
#cost_class1 = -labels_train * prediction
#cost_class2 = -(1-labels_train) * (1 - prediction)
#cost= cost_class1 + cost_class2
#labels_train = np.array(labels_train)

