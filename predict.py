import dill as pickle
import numpy as np

from cur_model import model_file

ask = 'y'
with open(model_file, 'rb') as f:
    model = pickle.load(f)
    while ask.lower() == 'y':
        y = model.predict([input("Type: "), input("GPA: "), input("SAT: "), input("ACT: "), input("Year: ")])
        print('Your getting', y[0], 'with a', int(np.round(y[1] * 100)), 'percent confidence')
        ask = input('Again? (Y/n): ')