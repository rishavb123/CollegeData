import dill as pickle
import numpy as np

from cur_model import find_model

ask = 'y'
while ask.lower() == 'y':
    model_file = find_model(input("College Name: "))
    if model_file == None:
        print("Invalid College Name")
        continue
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
        y = model.predict([input("Type: "), input("GPA: "), input("SAT: "), input("ACT: "), input("Year: ")])
        print('Your getting', y[0], 'with a', int(np.round(y[1] * 100)), 'percent confidence')
        ask = input('Again? (Y/n): ')
