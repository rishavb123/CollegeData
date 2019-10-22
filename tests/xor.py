import setup_imports
setup_imports.set_parent_path()

from nn import NeuralNetwork

inputs = [
    [[0], [0]],
    [[1], [1]],
    [[0], [1]],
    [[1], [0]]
]

targets = [
    [0],
    [0],
    [1],
    [1]
]

nn = NeuralNetwork([2, 4, 1])
nn.train_set(inputs, targets, epoch=10000)

for inp in inputs:
    print(inp[0], inp[1], ":", nn.predict(inp))