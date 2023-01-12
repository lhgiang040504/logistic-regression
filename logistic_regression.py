import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data_classification.csv", header=None)

x_true = []
y_true = []
x_false = []
y_false = []
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

def  sigmoid(z):
    return 1.0/(1 + np.exp(-1))

def filter_data(pred):
    if pred >= 0.5:
        return 1
    else:
        return 0
def predict(features, weights):
    pred = np.doc(features, weights)
    return sigmoid(pred)

def CE(features, weights, labels):
    '''matrics
    features: 100x3 (replace label with bias)
    weight: 1x3 (use the .T of numpy)
    label: 100x1
    '''
    prediction = predict(features, weights)
    n = len(df)
    cost_class1 = -labels*np.log(prediction)
    cost_class2 = -(1-labels)*np.log(1 - prediction)
    cost= cost_class1 + cost_class2
    return cost.sum()/n
