import os

model_file = "./models/model-tf-UMD--100-1573864974.pkl"


path = './models/'

models = []

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        models.append(os.path.join(r, file))

def find_model(name):
    name = name.replace("University of ", "U").replace("Technology", "Tech").replace("UIUC", "Illinois").replace("Maryland", "MD")

    for m in models:
        if m.split("-")[1].lower().replace(" ", "") == name.lower().replace(" ", "") or m.split("-")[2].lower().replace(" ", "") == name.lower().replace(" ", ""):
            return m
