import preprocessing as pp
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, nn, in_func, out_func):
        self.nn = nn
        self.in_func = in_func
        self.out_func = out_func

    def predict(self, inputs):
        return self.out_func(self.nn.predict(self.in_func(inputs)))

class TFModel:
    def __init__(self, nn, in_func, out_func):
        self.nn = None
        self.weights = nn.get_weights()
        self.in_func = lambda inps: np.array([pp.to_row(in_func(inps))])
        self.out_func = lambda inps: out_func(pp.to_column(inps[0]))
        self.input_dim = nn.layers[0].input_shape[1]
        self.output_dim = nn.layers[-1].output_shape[1]

    def predict(self, inputs):
        if self.nn == None:
            self.compile()
        return self.out_func(self.nn.predict(self.in_func(inputs)))

    def compile(self):
        self.nn = tf.keras.models.Sequential([
            tf.keras.layers.Dense(6, input_dim=self.input_dim, activation='sigmoid'),
            tf.keras.layers.Dense(6, activation='sigmoid'),
            tf.keras.layers.Dense(self.output_dim, activation='softmax'),
        ])
        self.nn.compile(optimizer='adam', loss='mse')
        self.nn.set_weights(self.weights)