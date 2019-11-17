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
    plt.xlabel("ACT")
    plt.ylabel("GPA")
    plt.title(college_name + " Scattergram")
    plt.scatter([max(sat_to_act(d[6]) if d[6].isdigit() else 0, int(d[7]) if d[7].isdigit() else 0) for d in dataset], [float(d[5]) for d in dataset], marker='x', c=['green' if d[1] in ['Accepted', 'Deferred'] else 'red' if d[1] in ['Denied', 'Waitlisted'] else 'yellow' for d in dataset])
    plt.show()