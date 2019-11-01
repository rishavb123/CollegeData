import csv
import matplotlib.pyplot as plt
import preprocessing as pp
from sat_to_act import sat_to_act

labels = None
dataset = None

college_name = 'UMD'

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
    
    results, in_func, out_func = pp.preprocess(dataset)

    plt.scatter([r[0][2] for r in results], [r[0][3] for r in results], c=['green' if r[1][1] == 1 else 'red' for r in results])
    plt.show()