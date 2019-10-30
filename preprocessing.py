from sat_to_act import sat_to_act
import numpy as np

def preprocess(data_orig):

    data = [[data_orig[x][y] for y in range(len(data_orig[0]))] for x in range(len(data_orig))]


    acts = []
    types = []
    results = []
    attends = []
    i = 0
    years = []
    while True:
        if i >= len(data):
            break
        d = data[i]
        years.append(int(d[8]))
        acts.append(max((sat_to_act(int(d[6])) if d[6].isdigit() else 0, int(d[7]) if d[7].isdigit() else 0)))
        if d[1] in ['Unknown', 'Incomplete', 'No decision', 'Guar. Transfer', 'Withdrawn'] or (d[6] == '-' and d[7] == '-'):
            del data[i]
            i -= 1
            continue
        if d[1] in ['Waitlisted', 'Deferred'] or 'Admit' in d[1] or 'Accept' in d[1]:
            data[i][1] = 'Accepted'
        if d[0] == "PRI":
            data[i][0] = "EA"
        if d[0] == "ROLL":
            data[i][0] = "RD"
        if d[0] not in types:
            types.append(d[0])
        if d[1] not in results:
            results.append(d[1])
        if d[4] not in attends:
            attends.append(d[4])
        i += 1

    result = []

    for d in data:
        del d[2:5]
        d[2] = float(d[2]) / 5
        d[3] = sat_to_act(d[3])
        d[3:5] = [norm(max((d[3], int(d[4]) if d[4].isdigit() else 0)), acts)]
        d[0:1] = one_hot(d[0], types)
        output = one_hot(d[2], results)
        del d[2]
        d[4] = int(d[4]) - min(years)
        result.append((d, output))

    def in_func(x_orig):
        x = one_hot(x_orig[0], types)
        x.append(float(x_orig[1]) / 5)
        x.append(norm(max((sat_to_act(int(x_orig[2])) if x_orig[2].isdigit() else 0, int(x_orig[3]) if x_orig[3].isdigit() else 0)), acts))
        x.append(int(x_orig[4]) - min(years))
        return to_column(x)

    def out_func(y_orig):
        y_orig = np.transpose(y_orig).tolist()[0]
        i = y_orig.index(max(y_orig))
        return results[i], soft_max(i, y_orig)

    for i in range(0, 37, 5):
        for j in np.arange(0, 5, 1):
            i = in_func([np.random.choice(types), str(j), '-', str(i), str(np.random.choice(years))])
            o = one_hot("Denied", results)
            result.append((i, o))

    count = np.zeros_like(result[0][1])

    for r in result:
        count += np.array(r[1])

    count = count.tolist()

    a = []
    i = 0
    while count.index(min(count)) != count.index(max(count)):
        if i >= len(result):
            i = 0
        if result[i][1].index(max(result[i][1])) == count.index(min(count)):
            count[count.index(min(count))] += 1
            a.append(result[i])
        i += 1

    result.extend(a)

    count = np.zeros_like(result[0][1])

    for r in result:
        count += np.array(r[1])

    count = count.tolist()
    print(count, results)

    return result, in_func, out_func

def norm(v, arr):
    return (v - min(arr)) / (max(arr) - min(arr))

def soft_max(i, arr):
    return np.exp(arr[i]) / (sum(np.exp(arr)) - 1)

def to_column(arr):
    return np.reshape(arr, (len(arr), 1))

def one_hot(a, arr):
    index = arr.index(a)
    a = []
    for i in range(len(arr)):
        if i == index:
            a.append(1)
        else:
            a.append(0)
    return a

def percentile(point, data_list):
    lis = data_list.copy()
    lis.sort()
    for i in range(len(lis)):
        if lis[i] > point:
            return i / (len(lis) + 1)
    return 1

