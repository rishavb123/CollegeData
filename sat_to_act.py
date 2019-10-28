f = open('./data/sat_to_act.txt')
arr = [[int(i) for i in a[:-1].split(" ")] for a in f.readlines()]
a = {}
for ar in arr:
    a[ar[0]] = ar[1]

def sat_to_act(sat):
    return a[int(sat)] if type(sat) == type(0) or sat.isdigit() else 0

# print(sat_to_act(500))