import os
from tqdm import tqdm
from natsort import natsorted
from pydub import silence, AudioSegment
from moviepy.editor import VideoFileClip


def preprocess_video():
    '''
    Takes all mp4 files in raw -> slices them by silence into chunks
    '''
    for file in tqdm(os.listdir('tts/data/dataset/raw')):
        print('Splitting Audio')
        if file.endswith('mp4'):
            split_audio_from_video(f'tts/data/dataset/raw/{file}', file.split(".")[0])


def split_audio_from_video(input_file, fname):
    clip = VideoFileClip(input_file)
    audio = clip.audio
    sr = audio.fps
    audio.write_audiofile('tts/data/dataset/raw/tmp.wav', fps=sr)
    audio = AudioSegment.from_wav('tts/data/dataset/raw/tmp.wav')    
    clips = split_audio_on_silence(audio, min_silence_len=250, fps=sr)
    os.makedirs(f'tts/data/dataset/raw/{fname}/audio', exist_ok=True)
    for i, clip in enumerate(clips):
        if clip.duration_seconds < 5.0:
            clip.export(f'tts/data/dataset/raw/{fname}/audio/{i}_{fname}.wav', format='wav') 
    os.remove('tts/data/dataset/raw/tmp.wav')
    

def split_audio_on_silence(audio, min_silence_len, fps):
    audio_clips = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=-40,
        keep_silence=True,
        seek_step=1000) # should be equivalent to 1 video frame)
    return audio_clips

