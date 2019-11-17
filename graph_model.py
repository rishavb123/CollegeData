import matplotlib.pyplot as plt
import dill as pickle
import numpy as np
import tensorflow as tf
import preprocessing as pp

from cur_model import model_file

# Graph model does not work for tf models

name = model_file.split("-")[1]

t = 'ED'
t2 = 'RD'
year = 2017
year2 = 2018
year3 = 2019
year4 = 2020
sat = '-'

with open(model_file, 'rb') as f:
    model = pickle.load(f)
    try:
        model.in_func([t, "0", sat, "0", year])
    except:
        t = 'EA'

    dataset = []
    for x in range(37):
        for y in np.arange(0, 5, 0.1):
            dataset.append([x, y])

    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    # plt.get_current_fig_manager().full_screen_toggle()
    fig.subplots_adjust(hspace=0.3)

    # col 1
    axs[0][0].scatter([d[0] for d in dataset], [d[1] for d in dataset], s=20, marker="s", c=['green' if model.predict([t, str(d[1]), sat, str(d[0]), str(year)])[0] == 'Accepted' else 'red' for d in dataset])
    axs[0][0].set_title(name + " " + t + " " + str(year))
    axs[0][0].set_xlabel('ACT (or Converted SAT)')
    axs[0][0].set_ylabel('GPA')
    
    axs[1][0].scatter([d[0] for d in dataset], [d[1] for d in dataset], s=20, marker="s", c=['green' if model.predict([t2, str(d[1]), sat, str(d[0]), str(year)])[0] == 'Accepted' else 'red' for d in dataset])
    axs[1][0].set_title(name + " " + t2 + " " + str(year))
    axs[1][0].set_xlabel('ACT (or Converted SAT)')
    axs[1][0].set_ylabel('GPA')
    
    # col 2
    axs[0][1].scatter([d[0] for d in dataset], [d[1] for d in dataset], s=20, marker="s", c=['green' if model.predict([t, str(d[1]), sat, str(d[0]), str(year2)])[0] == 'Accepted' else 'red' for d in dataset])
    axs[0][1].set_title(name + " " + t + " " + str(year2))
    axs[0][1].set_xlabel('ACT (or Converted SAT)')
    axs[0][1].set_ylabel('GPA')
    
    axs[1][1].scatter([d[0] for d in dataset], [d[1] for d in dataset], s=20, marker="s", c=['green' if model.predict([t2, str(d[1]), sat, str(d[0]), str(year2)])[0] == 'Accepted' else 'red' for d in dataset])
    axs[1][1].set_title(name + " " + t2 + " " + str(year2))
    axs[1][1].set_xlabel('ACT (or Converted SAT)')
    axs[1][1].set_ylabel('GPA')

    # col 3
    axs[0][2].scatter([d[0] for d in dataset], [d[1] for d in dataset], s=20, marker="s", c=['green' if model.predict([t, str(d[1]), sat, str(d[0]), str(year3)])[0] == 'Accepted' else 'red' for d in dataset])
    axs[0][2].set_title(name + " " + t + " " + str(year3))
    axs[0][2].set_xlabel('ACT (or Converted SAT)')
    axs[0][2].set_ylabel('GPA')
    
    axs[1][2].scatter([d[0] for d in dataset], [d[1] for d in dataset], s=20, marker="s", c=['green' if model.predict([t2, str(d[1]), sat, str(d[0]), str(year3)])[0] == 'Accepted' else 'red' for d in dataset])
    axs[1][2].set_title(name + " " + t2 + " " + str(year3))
    axs[1][2].set_xlabel('ACT (or Converted SAT)')
    axs[1][2].set_ylabel('GPA')

    # col 4
    axs[0][3].scatter([d[0] for d in dataset], [d[1] for d in dataset], s=20, marker="s", c=['green' if model.predict([t, str(d[1]), sat, str(d[0]), str(year4)])[0] == 'Accepted' else 'red' for d in dataset])
    axs[0][3].set_title(name + " " + t + " " + str(year4))
    axs[0][3].set_xlabel('ACT (or Converted SAT)')
    axs[0][3].set_ylabel('GPA')
    
    axs[1][3].scatter([d[0] for d in dataset], [d[1] for d in dataset], s=20, marker="s", c=['green' if model.predict([t2, str(d[1]), sat, str(d[0]), str(year4)])[0] == 'Accepted' else 'red' for d in dataset])
    axs[1][3].set_title(name + " " + t2 + " " + str(year4))
    axs[1][3].set_xlabel('ACT (or Converted SAT)')
    axs[1][3].set_ylabel('GPA')
    
    plt.show()