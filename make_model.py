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
learning_curve = True
use_tf = False
tag = ""

# Hyperparameters
epoch = 1000
hidden_layers = 2
hidden_nodes = 6
activation = 'sigmoid'
# alpha = 0.1

names = ['Cornell', 'GeorgiaTech', 'Illinois', 'UMD', 'UMich']
college_name = 'Illinois'

datasets = {}

load_data(college_name, datasets)

results, in_func, out_func = pp.preprocess(datasets[college_name])
input_dim = len(results[0][0])
output_dim = len(results[0][1])

train_data, cross_validation_data, test_data = split_data(results, validation_split=0)

nn = None

if learning_curve and not use_tf:
    nn = make_nn(hidden_layers, hidden_nodes, activation, input_dim, output_dim)
    costs, test_costs = nn.create_learning_curves([pp.to_column(r[0]) for r in train_data], [pp.to_column(r[1]) for r in train_data], [pp.to_column(r[0]) for r in test_data], [pp.to_column(r[1]) for r in test_data], epoch=epoch)
    print(nn.evaluate_classification([pp.to_column(r[0]) for r in test_data], [pp.to_column(r[1]) for r in test_data]))
    x = range(epoch)
    y = costs
    test_y = test_costs

    def smooth(y, ave_window_size=10, savgol_window_size=epoch/2):
        run_ave = [sum(y[0:i]) / i for i in range(1, 1 + len(y))]
        loc_ave = [sum(y[i-ave_window_size if i >= ave_window_size else 0:i]) / (i - (i-ave_window_size if i >= ave_window_size else 0)) for i in range(1, 1 + len(y))]
        y_smooth = savgol_filter(y, savgol_window_size, 3)
        return run_ave, loc_ave, y_smooth

    run_ave, loc_ave, y_smooth = smooth(y)
    test_run_ave, test_loc_ave, test_y_smooth = smooth(test_y)

    # plt.plot(x, y, label="Original")
    # plt.plot(x, run_ave, label="Running / Moving Average")
    # plt.plot(x, loc_ave, label="Local Average")
    plt.plot(x, y_smooth, label="Savitzky–Golay Smoothing Filter")

    # plt.plot(x, test_y, label="Test Original")
    # plt.plot(x, test_run_ave, label="Test Running / Moving Average")
    # plt.plot(x, test_loc_ave, label="Test Local Average")
    plt.plot(x, test_y_smooth, label="Test Savitzky–Golay Smoothing Filter")

    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()
else:

    if use_tf:
        nn = make_tf_nn(hidden_layers, hidden_nodes, activation, input_dim, output_dim)
        nn.fit(np.array([pp.to_row(r[0]) for r in train_data], "float32"), np.array([pp.to_row(r[1]) for r in train_data], "float32"), epochs=epoch)
        print(nn.evaluate(np.array([pp.to_row(r[0]) for r in test_data], "float32"), np.array([pp.to_row(r[1]) for r in test_data], "float32")))
    else:
        nn = make_nn(hidden_layers, hidden_nodes, activation, input_dim, output_dim)
        costs = nn.train_set([pp.to_column(r[0]) for r in train_data], [pp.to_column(r[1]) for r in train_data], epoch, plot)
        print(nn.evaluate_classification([pp.to_column(r[0]) for r in test_data], [pp.to_column(r[1]) for r in test_data]))

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