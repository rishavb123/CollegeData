import csv
import matplotlib.pyplot as plt
from sat_to_act import sat_to_act

labels = None
dataset = None

college_name = 'Illinois'

with open('./data/College Data - ' + college_name + '.csv') as f:
    data = csv.reader(f)
    # labels = data[0]
    # dataset = data[1:]
    dataset = []
    i = 0
    for d in data:
        if i != 0:
            dataset.append(d)
        i += 1
    sats = []
    acts = []
    gpas = []
    fig = plt
    for d in dataset:
        gpas.append(float(d[5]))
        if d[6].isdigit():
            sats.append(int(d[6]))
        if d[7].isdigit():
            acts.append(int(d[7]))
    print(min(sats), sum(sats) / len(sats), max(sats), max(set(sats), key=sats.count))
    print(min(acts), sum(acts) / len(acts), max(acts), max(set(acts), key=acts.count))
    print(min(gpas), sum(gpas) / len(gpas), max(gpas), max(set(gpas), key=gpas.count))

    fig, axs = plt.subplots(1, 3, figsize=(15, 7))
    axs[0].hist(sats)
    axs[0].set_title(college_name + " SATs")
    axs[1].hist(acts)
    axs[1].set_title(college_name + " ACTs")
    axs[2].hist([sat_to_act(sat) for sat in sats] + acts)
    axs[2].set_title(college_name + " SATs and ACTs")
    plt.show()