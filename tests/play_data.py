import csv

labels = None
dataset = None

college_name = 'Cornell'

with open('../data/College Data - ' + college_name + '.csv') as f:
    data = csv.reader(f)
    labels = data[0]
    dataset = data[1:]