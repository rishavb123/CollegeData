import csv
import dill as pickle
import time
import preprocessing as pp
from nn import NeuralNetwork
import numpy as np
# import json

labels = None
dataset = None

epoch = 100
college_name = 'Cornell'
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
nn.train_set([pp.to_column(r[0]) for r in results], [pp.to_column(r[1]) for r in results], epoch)

class Model:
    def __init__(self, nn, in_func, out_func):
        self.nn = nn
        self.in_func = in_func
        self.out_func = out_func

    def predict(self, inputs):
        return self.out_func(self.nn.predict(self.in_func(inputs)))

model = Model(nn, in_func, out_func)
with open('./models/model-' + college_name + "-" + ((tag + "-") if tag != None else "") + str(epoch) + "-" + str(int(time.time())) + '.pkl', 'wb') as f:
    pickle.dump(model, f) 
        