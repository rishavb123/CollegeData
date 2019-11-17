import csv
import matplotlib.pyplot as plt
import preprocessing as pp
from sat_to_act import sat_to_act

labels = None
dataset = None

college_name = 'Illinois'

with open('./data/College Data - ' + college_name + '.csv') as f:
    data = csv.reader(f)
    dataset = []
    i = 0
    for d in data:
        if i != 0:
            dataset.append(d)
        i += 1
    
    results, in_func, out_func = pp.preprocess(dataset)

    plt.scatter([r[0][2] for r in results], [r[0][3] for r in results], c=['green' if r[1][1] == 1 else 'red' for r in results])
    plt.show()