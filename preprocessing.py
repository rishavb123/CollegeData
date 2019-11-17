from sat_to_act import sat_to_act
import numpy as np

def preprocess(data_orig):

    data = [[data_orig[x][y] for y in range(len(data_orig[0]))] for x in range(len(data_orig))]


    acts = []
    gpas = []
    parallel_results = []

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
        
        if d[1] in ['Unknown', 'Incomplete', 'No decision', 'Guar. Transfer', 'Withdrawn'] or (d[6] == '-' and d[7] == '-'):
            del data[i]
            i -= 1
            continue
        if d[1] in ['Waitlisted']:
            data[i][1] = 'Denied'
        if d[1] in ['Deferred'] or 'Admit' in d[1] or 'Accept' in d[1]:
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

        gpas.append(float(d[5]))
        acts.append(max((sat_to_act(int(d[6])) if d[6].isdigit() else 0, int(d[7]) if d[7].isdigit() else 0)))
        parallel_results.append(data[i][1])

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
        if x_orig[0] == "PRI":
            x_orig[0] = "EA"
        elif x_orig[0] == "ROLL":
            x_orig[0] = "RD"
        x = one_hot(x_orig[0], types)
        x.append(float(x_orig[1]) / 5)
        x.append(norm(max((sat_to_act(int(x_orig[2])) if x_orig[2].isdigit() else 0, int(x_orig[3]) if x_orig[3].isdigit() else 0)), acts))
        x.append(int(x_orig[4]) - min(years))
        return to_column(x)

    def out_func(y_orig):
        y_orig = np.transpose(y_orig).tolist()[0]
        i = y_orig.index(max(y_orig))
        return results[i], soft_max(i, y_orig)

    gpas = np.array(gpas)
    acts = np.array(acts)
    parallel_results = np.array(parallel_results)

    for i in range(37):
        for j in range(6):
            inp = in_func([np.random.choice(types), str(j), '-', str(i), str(np.random.choice(years))])
            r = "Denied"
            r1 = np.where(gpas == min(gpas, key=lambda x: np.abs(j-x)))
            r2 = np.where(acts == min(acts, key=lambda x: np.abs(i-x)))
            # print('-------------------------------------------------------------------')
            # print("GPA", j, r1, gpas[r1[0]], acts[r1[0]])
            # print("ACT", i, r2, acts[r2[0]])
            # # a1, b1 = np.unique(parallel_results[r1], return_counts=True)
            # # a2, b2 = np.unique(parallel_results[r2], return_counts=True)
            dist1 = np.abs(i - acts[r1])
            dist2 = np.abs(j - gpas[r2])
            # print("distances(in act) gpa", dist1)
            # print("distances(in gpa) act", dist2)
            # print("Results GPA", parallel_results[r1][np.argmin(dist1)])
            # print("Results ACT", parallel_results[r2][np.argmin(dist2)])
            if parallel_results[r1][np.argmin(dist1)] == "Accepted" and parallel_results[r2][np.argmin(dist2)] == "Accepted":
                r = "Accepted"
            out = one_hot(r, results)
            result.append((inp, out))

    count = np.zeros_like(result[0][1])

    for r in result:
        count += np.array(r[1])

    count = count.tolist()

    a = []
    i = 0
    while min(count) != max(count):
        if i >= len(result):
            i = 0
        if result[i][1].index(max(result[i][1])) == count.index(min(count)):
            count[count.index(min(count))] += 1
            a.append(result[i])
        i += 1

    result.extend(a)

    return result, in_func, out_func

def norm(v, arr):
    return (v - min(arr)) / (max(arr) - min(arr))

def soft_max(i, arr):
    return np.exp(arr[i]) / (sum(np.exp(arr)) - 1)

def to_column(arr):
    return np.reshape(arr, (len(arr), 1))

def to_row(arr):
    return np.reshape(arr, (len(arr)))

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

