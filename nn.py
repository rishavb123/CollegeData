import numpy as np

class NeuralNetwork:
    
    relu = lambda x: x * (x > 0)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def __init__(self, shape, activation=sigmoid, learning_rate=0.1):
        self.shape = shape
        self.activation = np.vectorize(activation)
        self.activation_derivative = np.vectorize(NeuralNetwork.derivative(activation))

        self.weights = []
        self.biases = []

        for i in range(len(self.shape) - 1):
            self.weights.append(np.random.random([self.shape[i + 1], self.shape[i]]))
            self.biases.append(np.random.random([self.shape[i + 1], 1]))

        self.learning_rate = learning_rate

    def predict(self, inp):
        a = np.array(inp)
        for weight, bias in zip(self.weights, self.biases):
            a = self.activation(np.matmul(weight, a) + bias)
        return a

    def train(self, inp, target):
        target = np.array(target)
        a = [np.array(inp)]
        z = [a[0]]
        i = 0
        for weight, bias in zip(self.weights, self.biases):
            i += 1
            z.append(np.matmul(weight, a[len(a) - 1]) + bias)
            a.append(self.activation(z[len(z) - 1]))
        output = a[len(a) - 1]
        errors = [None for _ in range(len(self.shape))]
        errors[len(errors) - 1] = target - output

        delta_weights = [None for _ in self.weights]
        delta_biases = [None for _ in self.biases]

        for i in range(len(self.weights), 0, -1):
            errors[i - 1] = np.matmul(np.transpose(self.weights[i - 1]), errors[i])
            temp = errors[i] * self.activation_derivative(z[i])
            delta_weights[i - 1]= np.matmul(temp, np.transpose(a[i - 1])) * self.learning_rate
            delta_biases[i - 1] = temp * self.learning_rate
            self.weights[i - 1] += delta_weights[i - 1]
            self.biases[i - 1] += delta_biases[i - 1]

    def predict_set(self, inputs):
        return [self.predict(inp) for inp in inputs]

    def train_set(self, inputs, targets, epoch=1):
        percent = 0
        for i in range(epoch):
            while i / epoch > percent:
                print(int(np.round(percent * 100)), "percent done")
                percent += .1
            inputs, targets = NeuralNetwork.shuffle(inputs, targets)
            for inp, target in zip(inputs, targets):
                self.train(inp, target)
        print("100 percent done")

    @staticmethod
    def derivative(f, h=0.0001):
        return lambda x: (f(x + h / 2) - f(x - h / 2)) / h

    @staticmethod
    def shuffle(a, b):
        a = np.array(a)
        b = np.array(b)
        p = np.random.permutation(len(a))
        return a[p].tolist(), b[p].tolist()