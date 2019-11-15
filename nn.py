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
            self.weights.append(np.random.random([self.shape[i + 1], self.shape[i]]) * 2 - 1)
            self.biases.append(np.random.random([self.shape[i + 1], 1]) * 2 - 1)

        self.learning_rate = learning_rate

    def predict(self, inp):
        a = np.array(inp)
        for weight, bias in zip(self.weights, self.biases):
            a = self.activation(np.matmul(weight, a) + bias)
        return a

    def train(self, inp, target):
        loss = []

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

    def train_set(self, inputs, targets, epoch=1, with_cost=False):
        percent = 0
        costs = []
        for i in range(epoch):
            while i / epoch > percent:
                print(int(np.round(percent * 100)), "percent done")
                percent += .1
            inputs, targets = NeuralNetwork.shuffle(inputs, targets)
            for inp, target in zip(inputs, targets):
                self.train(inp, target)
            costs.append(self.cost(inputs, targets))
        print("100 percent done")
        return costs if len(costs) > 0 else None
    
    def loss(self, inp, target):
        return sum((self.predict(inp) - target) ** 2)

    def cost(self, inputs, targets):
        return sum([self.loss(inp, target) for inp, target in zip(inputs, targets)]) / len(inputs)

    def mutate(self, mutation_rate):
        for weight, bias in zip(self.weights, self.biases):
            for i in range(len(weight)):
                for j in range(len(weight[i])):
                    if np.random.random() < mutation_rate:
                        weight[i][j] = weight[i][j] + np.random.random() * 0.5 - 0.25

            for i in range(len(bias)):
                if np.random.random() < mutation_rate:
                    bias[i] = np.random.random() * 2 - 1
        return self

    @staticmethod
    def derivative(f, h=0.0001):
        return lambda x: (f(x + h / 2) - f(x - h / 2)) / h

    @staticmethod
    def shuffle(a, b):
        a = np.array(a)
        b = np.array(b)
        p = np.random.permutation(len(a))
        return a[p].tolist(), b[p].tolist()

    @staticmethod
    def crossover(nn1, nn2):
        if nn1.shape != nn2.shape:
            return None
        nn = NeuralNetwork(nn1.shape)
        for i in range(len(nn1.weights)):
            nn.weights[i] = np.add(nn1.weights[i], nn2.weights[i]) / 2
            nn.biases[i] = np.add(nn1.biases[i], nn2.biases[i]) / 2
        return nn

    @staticmethod
    def clone(nn):
        nn2 = NeuralNetwork(nn.shape)
        for i in range(len(nn.weights)):
            nn2.weights[i] = nn.weights[i].copy()
            nn2.biases[i] = nn.biases[i].copy()
        return nn