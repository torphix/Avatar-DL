import json
import shutil

def make_dataset_file(root_path, dataset_name):
    with open(f'{root_path}/text/jocko.txt') as f:
        text = f.readlines()
    # Preprocess
    text = [t.strip('\n') for t in text]
    text = [t.split('|')[0] for t in text]
    text = [t.strip(' ') for t in text]
    data = {"root_path":f'{root_path}/processed',}
    samples = []
    for i, t in enumerate(text):
        with open(f'{root_path}/processed/{i+1}_jocko.txt', 'w') as f:
            f.write(t)
        samples.append({
            "transcript_path": f'{i+1}_jocko.txt',
            "wav_path": f"{i+1}_jocko.wav"
        })
        shutil.copy(f'{root_path}/audio/{i+1}_jocko.wav', f'{root_path}/processed/{i+1}_jocko.wav')
    data['samples'] = samples
    with open(f'{root_path}/{dataset_name}.json', 'w') as f:
        f.write(json.dumps(data, indent=4))
    
    
if __name__ == '__main__':
    make_dataset_file('/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/stt/data','jocko')