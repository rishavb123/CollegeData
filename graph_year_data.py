import csv
import matplotlib.pyplot as plt
from sat_to_act import sat_to_act
import numpy as np

labels = None
dataset = None

college_name = 'UMich'

with open('./data/College Data - ' + college_name + '.csv') as f:
    data = csv.reader(f)
    dataset = []
    i = 0
    for d in data:
        if i != 0:
            dataset.append(d)
        i += 1
    i = 0
    while i < len(dataset):
        d = dataset[i]
        if not dataset[i][1] in ['Accepted', 'Denied', 'Deferred', 'Waitlisted']:
            dataset.remove(d)
            i-=1
        elif max(sat_to_act(d[6]) if d[6].isdigit() else 0, int(d[7]) if d[7].isdigit() else 0) == 0:
            dataset.remove(d)
            i-=1
        i+=1
    dataset = np.array(dataset)
    
    fig, axs = plt.subplots(1, 4, figsize=(15, 7))
    # plt.get_current_fig_manager().full_screen_toggle()
    fig.subplots_adjust(hspace=0.3)

    for i in range(4):
        axs[i].set_xlabel("ACT")
        axs[i].set_ylabel("GPA")
        year = i + 2016
        axs[i].set_title(college_name + " " + str(year) + " Scattergram")
        year_dataset = dataset[[int(row[8]) == year for row in dataset]]
        axs[i].scatter([max(sat_to_act(d[6]) if d[6].isdigit() else 0, int(d[7]) if d[7].isdigit() else 0) for d in year_dataset], [float(d[5]) for d in year_dataset], marker='x', c=['green' if d[1] in ['Accepted', 'Deferred'] else 'red' if d[1] in ['Denied', 'Waitlisted'] else 'yellow' for d in year_dataset])
    plt.show()