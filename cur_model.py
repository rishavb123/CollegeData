# model_file = "./models/model-GeorgiaTech-100-1572632288.pkl"
# model_file = "./models/model-UMD-100-1572631031.pkl"
# model_file = "./models/model-Cornell--100-1573586686.pkl"
# model_file = "./models/model-Illinois--100-1573757349.pkl"
# model_file = "./models/model-UMich--100-1573758017.pkl"
models = ["./models/model-Cornell--100-1573586686.pkl", "./models/model-GeorgiaTech-100-1572632288.pkl", "./models/model-Illinois--100-1573757349.pkl", "./models/model-UMD-100-1572631031.pkl", "./models/model-UMich--100-1573758017.pkl"]
model_file = models[-1]

def find_model(name):
    name = name.replace("University of ", "U").replace("Technology", "Tech").replace("UIUC", "Illinois").replace("Maryland", "MD")

    for m in models:
        if m.split("-")[1].lower().replace(" ", "") == name.lower().replace(" ", ""):
            return m
