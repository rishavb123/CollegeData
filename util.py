import csv
import numpy as np
import tensorflow as tf
from nn import NeuralNetwork

activations = ['tanh', 'sigmoid', 'relu', 'linear']

def load_data(cur_name, datasets):
    with open('./data/College Data - ' + cur_name + '.csv') as f:
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
        datasets[cur_name] = dataset

def split_data(data, validation_split=0.1, test_split=0.1):
    np.random.shuffle(data)
    cross_validation_data = data[:int(len(data) * validation_split)]
    test_data = data[int(len(data) * validation_split) : int(len(data) * (validation_split + test_split))]
    train_data = data[int(len(data)  * (validation_split + test_split)):]
    return train_data, cross_validation_data, test_data

def unpack_dna(dna):
    epoch = dna[0] * 10
    hidden_layers = dna[1]
    hidden_nodes = dna[2]
    activation = activations[dna[3]]
    return epoch, hidden_layers, hidden_nodes, activation

def dict_dna(dna):
    epoch, hidden_layers, hidden_nodes, activation = unpack_dna(dna)
    return {"epoch": epoch, "hidden_layers": hidden_layers, "hidden_nodes": hidden_nodes, "activation": activation}

def make_tf_nn(hidden_layers, hidden_nodes, activation, input_dim, output_dim):
    nn = tf.keras.models.Sequential()
    first_layer = True
    for _ in range(hidden_layers):
        if first_layer:
            nn.add(tf.keras.layers.Dense(hidden_nodes, input_dim=input_dim, activation=activation))
            first_layer = False
        else:
            nn.add(tf.keras.layers.Dense(hidden_nodes, activation=activation))
    if first_layer:
        nn.add(tf.keras.layers.Dense(output_dim, input_dim=input_dim, activation='softmax'))
        first_layer = False
    else:
        nn.add(tf.keras.layers.Dense(output_dim, activation='softmax'))
    nn.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return nn

def make_tf_nn_with_dna(dna, input_dim, output_dim):
    epoch, hidden_layers, hidden_nodes, activation = unpack_dna(dna)
    return make_tf_nn(hidden_layers, hidden_nodes, activation, input_dim, output_dim)

def make_nn(hidden_layers, hidden_nodes, activation, input_dim, output_dim):
    shape = [hidden_nodes for _ in range(hidden_layers)]
    shape.append(output_dim)
    shape.insert(0, input_dim)
    return NeuralNetwork(shape, activation=activation)

def make_nn_with_dna(dna, input_dim, output_dim):
    epoch, hidden_layers, hidden_nodes, activation = unpack_dna(dna)
    return make_nn(hidden_layers, hidden_nodes, activation, input_dim, output_dim)

def clamp(x, r):
    return x if r[0] <= x <= r[1] else (r[0] if x < r[0] else r[1])
