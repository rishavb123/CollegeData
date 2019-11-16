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
    [[0], [1]],
    [[0], [1]],
    [[1], [0]],
    [[1], [0]]
]

nn = NeuralNetwork([2, 4, 2], activation='relu')
nn.train_set(inputs, targets, epoch=10000)

for inp, tar in zip(inputs, targets):
    out = nn.predict(inp)
    print(inp[0], inp[1], ":", tar[0], tar[1], ":", out[0], out[1])

print(nn.evaluate_classification(inputs, targets))