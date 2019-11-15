import csv
import dill as pickle
import time
import preprocessing as pp
import matplotlib.pyplot as plt
from nn import NeuralNetwork
import numpy as np
from savgol_filter import savgol_filter

labels = None
dataset = None

save = True
plot = True

epoch = 100
college_name = 'Illinois'
tag = ""

names = ['Cornell', 'GeorgiaTech', 'Illinois', 'UMD', 'UMich']


datasets = {}
labels = {}


# for college_name in names:
with open('./data/College Data - ' + college_name + '.csv') as f:
    data = csv.reader(f)
    dataset = []
    label = []
    i = 0
    for row in data:
        if i == 0:
            label = row
        else:
            dataset.append(row)
        i += 1
    labels[college_name] = label
    datasets[college_name] = dataset

# labels = labels[college_name]
# del labels[2:4]
# print(labels)

results, in_func, out_func = pp.preprocess(datasets[college_name])


nn = NeuralNetwork([len(results[0][0]), 6, 6, len(results[0][1])])
costs = nn.train_set([pp.to_column(r[0]) for r in results], [pp.to_column(r[1]) for r in results], epoch, plot)

if plot:
    x = range(epoch)
    y = costs
    run_ave = [sum(y[0:i]) / i for i in range(1, 1 + len(y))]
    loc_ave = [sum(y[i-5 if i >= 5 else 0:i]) / (i - (i-5 if i >= 5 else 0)) for i in range(1, 1 + len(y))]
    y_smooth = savgol_filter(y, epoch / 2, 3)

    plt.plot(x, y, label="Original")
    plt.plot(x, run_ave, label="Running / Moving Average")
    plt.plot(x, loc_ave, label="Local Average")
    plt.plot(x, y_smooth, label="Savitzky–Golay Smoothing Filter")

    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

class Model:
    def __init__(self, nn, in_func, out_func):
        self.nn = nn
        self.in_func = in_func
        self.out_func = out_func

    def predict(self, inputs):
        return self.out_func(self.nn.predict(self.in_func(inputs)))

if save:
    model = Model(nn, in_func, out_func)
    with open('./models/model-' + college_name + "-" + ((tag + "-") if tag != None else "") + str(epoch) + "-" + str(int(time.time())) + '.pkl', 'wb') as f:
        pickle.dump(model, f) 
        