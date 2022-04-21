import json
import shutil

with open(f'/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/stt/data/datasets/processed/dataset.json', 'r') as f:
    data = f.readlines()

for datapoint in data:
    datapoint = json.loads(datapoint.strip("\n"))
    text = datapoint['transcription']
    print(datapoint)
    shutil.copy(datapoint['path'], f'/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/data/datasets/lex_fridman/hand_labelled/{datapoint["path"].split("/")[-1]}')
    with open(f'/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/data/datasets/lex_fridman/hand_labelled/{datapoint["path"].split("/")[-1].split(".")[0]}.lab', 'w') as f:
        f.write(text)
    