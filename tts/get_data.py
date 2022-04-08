import os
import torch
import librosa
from tqdm import tqdm
from stt.asr import ASR
from pytube import YouTube
from pydub import AudioSegment, silence
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


class YoutubeTTSData:
    def __init__(self, 
                 links,
                 output_path,
                 min_silence_len,
                 seperate_speakers):
        super().__init__()
        self.links = links
        self.output_path = output_path
        self.min_silence_len = min_silence_len
        
        self.seperate_speakers = seperate_speakers
        if self.seperate_speakers:
            self.speaker_seperator = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
            
        self.asr = ASR()            

    def get_audio_data(self):
        clip_paths = []
        for link in tqdm(self.links):
            yt=YouTube(link)
            t = yt.streams.filter(only_audio=True).first()
            new_title = yt.title.split(':')[0].replace(' ', '_').lower().split(".")[0]
            os.makedirs(f'{self.output_path}/Audio/{new_title}', exist_ok=True)
            save_path = f'{self.output_path}/{new_title}.mp3'
            t.download(self.output_path, f'{new_title}.mp3')
            audio = AudioSegment.from_file(save_path)
            audio = audio.set_channels(1)
            audio_clips = silence.split_on_silence(audio,
                                                   min_silence_len=self.min_silence_len,
                                                   silence_thresh=-40,
                                                   keep_silence=True) # should be equivalent to 1 video frame)
            for i, clip in enumerate(audio_clips):
                clip.export(f'{self.output_path}/Audio/{new_title}/{i}_{new_title}.mp3') 
                clip_paths.append(f'{self.output_path}/Audio/{new_title}/{i}_{new_title}.mp3')
            os.remove(save_path)            
        if self.seperate_speakers:
            for path in clip_paths:
                audio, sr = librosa.load(path, 16000, mono=True)
                inputs = self.feature_extractor(audio,
                                                sampling_rate=16000,
                                                padding=True,
                                                return_tensors="pt")
                logits = self.speaker_seperator(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                print(predicted_ids)

        os.makedirs(f'{self.output_path}/Text', exist_ok=True)
        for clip in clip_paths:
            f_name = clip.split("/")[-1].split(".")[0]
            self.asr.file_inference(clip, f'{self.output_path}/Text/{new_title}/{f_name}.txt')



def get_data(
    links, output_path, min_silence_len, seperate_speakers):
    dataset_generator = YoutubeTTSData(
        links,
        output_path,
        min_silence_len,
        seperate_speakers
    )
    dataset_generator.get_audio_data()