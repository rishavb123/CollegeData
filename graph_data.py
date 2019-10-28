import csv
import matplotlib.pyplot as plt
from sat_to_act import sat_to_act

labels = None
dataset = None

college_name = 'GeorgiaTech'

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
    # fig, axs = plt.subplots(1, 4, figsize=(20, 10))
    # axs[0].hist(sats)
    # axs[1].hist(acts)
    # axs[2].hist([sat_to_act(sat) for sat in sats] + acts)
    # axs[3].scatter([float(d[5]) for d in dataset], [max(sat_to_act(d[6]) if d[6].isdigit() else 0, int(d[7]) if d[7].isdigit() else 0) for d in dataset], c=['green' if d[1] in ['Accepted', 'Waitlisted', 'Deferred'] else 'red' if d[1] == 'Denied' else 'yellow' for d in dataset])
    
    # plt.hist(sats)
    # plt.hist(acts)
    # plt.hist([sat_to_act(sat) for sat in sats] + acts)
    plt.scatter([float(d[5]) for d in dataset], [max(sat_to_act(d[6]) if d[6].isdigit() else 0, int(d[7]) if d[7].isdigit() else 0) for d in dataset], c=['green' if d[1] in ['Accepted', 'Waitlisted', 'Deferred'] else 'red' if d[1] == 'Denied' else 'yellow' for d in dataset])
    plt.show()