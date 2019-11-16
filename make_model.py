import dill as pickle
import time
import preprocessing as pp
import matplotlib.pyplot as plt
from nn import NeuralNetwork
import numpy as np
from savgol_filter import savgol_filter
import tensorflow as tf

from util import *
from wrappers import *

labels = None
dataset = None


# Config
save = False
plot = True
use_tf = False
tag = ""

# Hyperparameters
epoch = 10
hidden_layers = 2
hidden_nodes = 6

names = ['Cornell', 'GeorgiaTech', 'Illinois', 'UMD', 'UMich']
college_name = 'UMD'

datasets = {}

load_data(college_name, datasets)

results, in_func, out_func = pp.preprocess(datasets[college_name])

train_data, cross_validation_data, test_data = split_data(results)

nn = None

if use_tf:
    nn = make_tf_nn(hidden_layers, hidden_nodes, len(results[0][0]), len(results[0][1]))
    nn.fit(np.array([pp.to_row(r[0]) for r in train_data], "float32"), np.array([pp.to_row(r[1]) for r in train_data], "float32"), epochs=epoch)
    print(nn.evaluate(np.array([pp.to_row(r[0]) for r in cross_validation_data], "float32"), np.array([pp.to_row(r[1]) for r in cross_validation_data], "float32")))
else:
    nn = make_nn(hidden_layers, hidden_nodes, len(results[0][0]), len(results[0][1]))
    costs = nn.train_set([pp.to_column(r[0]) for r in train_data], [pp.to_column(r[1]) for r in train_data], epoch, plot)
    print(nn.evaluate_classification([pp.to_column(r[0]) for r in cross_validation_data], [pp.to_column(r[1]) for r in cross_validation_data]))

if plot and not use_tf:
    x = range(epoch)
    y = costs
    run_ave = [sum(y[0:i]) / i for i in range(1, 1 + len(y))]
    loc_ave = [sum(y[i-10 if i >= 10 else 0:i]) / (i - (i-10 if i >= 10 else 0)) for i in range(1, 1 + len(y))]
    y_smooth = savgol_filter(y, epoch / 2, 3)

    plt.plot(x, y, label="Original")
    plt.plot(x, run_ave, label="Running / Moving Average")
    plt.plot(x, loc_ave, label="Local Average")
    plt.plot(x, y_smooth, label="Savitzky–Golay Smoothing Filter")

    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

model = Model(nn, in_func, out_func) if not use_tf else TFModel(nn, in_func, out_func)
if save:
    with open('./models/model' + ('-tf-' if use_tf else '-') + college_name + "-" + ((tag + "-") if tag != None else "") + str(epoch) + "-" + str(int(time.time())) + '.pkl', 'wb') as f:
        pickle.dump(model, f)