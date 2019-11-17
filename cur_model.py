import os



path = './models/'
optimized_path = './optimized_models/'

models = []
optimized_models = []

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        models.append(os.path.join(r, file))

for r, d, f in os.walk(optimized_path):
    for file in f:
        optimized_models.append(os.path.join(r, file))

model_file = "./optimized_models/model-GeorgiaTech-92-18, 1, 2, 1-1574021463.pkl"


def find_model(name):
    name = name.replace("University of ", "U").replace("Technology", "Tech").replace("UIUC", "Illinois").replace("Maryland", "MD")

    for m in optimized_models:
        if m.split("-")[1].lower().replace(" ", "") == name.lower().replace(" ", "") or m.split("-")[2].lower().replace(" ", "") == name.lower().replace(" ", ""):
            return m

    for m in models:
        if m.split("-")[1].lower().replace(" ", "") == name.lower().replace(" ", "") or m.split("-")[2].lower().replace(" ", "") == name.lower().replace(" ", ""):
            return m
