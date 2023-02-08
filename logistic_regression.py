import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("data_classification.csv", header=None)

train, test = train_test_split(df, test_size = 0.3, random_state=40)

x_train = df.drop(columns = df.columns[-1])
y_train = df[df.columns[-1]]

x_test = df.drop(columns = df.columns[-1])
y_test = df[df.columns[-1]]


'''
x_true = []
y_true = []
x_false = []
y_false = []
##1x100

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
    return 1.0/(1 + np.exp(-1))

def filter_data(pred):
    if pred >= 0.8:
        return 1
    else:
        return 0
def predict(features, weights):
    pred = np.dot(features, weights)
    return sigmoid(pred)

def CE(features, weights, labels):
    '''matrics
    features: 100x3 (replace label with bias)
    weight: 1x3 (use the .T of numpy)**
    label: 100x1
    '''
    prediction = predict(features, weights)
    n = len(df)
    cost_class1 = -labels*np.log(prediction)
    cost_class2 = -(1-labels)*np.log(1 - prediction)
    cost= cost_class1 + cost_class2
    return cost.sum()/n

def update_weights(features, weights, labels, learning_rate):
    '''matrics
    features: 100x3 (replace label with bias)
    weight: 1x3 (use the .T of numpy)**
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


weights, cost_history = train(features, weights, labels, 0.01, 100)