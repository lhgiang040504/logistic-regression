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