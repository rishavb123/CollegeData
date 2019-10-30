import matplotlib.pyplot as plt
import dill as pickle
import numpy as np

from cur_model import model_file

t = 'EA'
t2 = 'RD'
year = 2019
year2 = 2020
sat = '-'

with open(model_file, 'rb') as f:
    model = pickle.load(f)

    dataset = []
    for x in range(37):
        for y in np.arange(0, 5, 0.01):
            dataset.append([x, y])

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    axs[0][0].scatter([d[0] for d in dataset], [d[1] for d in dataset], c=['green' if model.predict([t, str(d[1]), sat, str(d[0]), str(year)])[0] == 'Accepted' else 'red' for d in dataset])
    axs[0][1].scatter([d[0] for d in dataset], [d[1] for d in dataset], c=['green' if model.predict([t2, str(d[1]), sat, str(d[0]), str(year)])[0] == 'Accepted' else 'red' for d in dataset])
    axs[1][0].scatter([d[0] for d in dataset], [d[1] for d in dataset], c=['green' if model.predict([t, str(d[1]), sat, str(d[0]), str(year2)])[0] == 'Accepted' else 'red' for d in dataset])
    axs[1][1].scatter([d[0] for d in dataset], [d[1] for d in dataset], c=['green' if model.predict([t2, str(d[1]), sat, str(d[0]), str(year2)])[0] == 'Accepted' else 'red' for d in dataset])
    plt.show()