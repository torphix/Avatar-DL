import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from pytube import YouTube
from natsort import natsorted
from pydub import AudioSegment
from pyannote.audio import Pipeline
from ..utils import split_video_py
from moviepy.editor import VideoFileClip


def create_lex_dataset(config):
    data_handler = LexDataHandler(config)
    data_handler.start()

class LexDataHandler(object):
    def __init__(self, processing_config):
        '''
        links: Youtube links
        silence_length_crop: in milli seconds
        max_frames: if memory contraint is a problem -1 for no limit
        max_clip_len: removes clips & audio if len is >= in ms
        min_clip_len: removes clips & audio if len is <= in ms
        '''
        self.links=processing_config['links']
        self.silence_length_crop=processing_config['silence_length_crop']
        self.silence_db_threshold=processing_config['silence_db_threshold']
        self.video_fps=processing_config['video_fps']
        self.min_clip_len=processing_config['min_clip_len']
        self.max_clip_len=processing_config['max_clip_len']
        self.save_path=processing_config['save_path']

        self.diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    def start(self):
        for link in tqdm(self.links):
            path, title = self.start_download(link)
            # Split video into chunks
            video_save_dir = f'{"/".join(path.split("/")[:-1])}/VideoFlash/{title}'
            audio_save_dir = f'{"/".join(path.split("/")[:-1])}/AudioWAV/{title}'
            tmp_path = self.get_intro(path)    
            split_video_py(title, 
                            f'{tmp_path}/tmp.wav',
                            f'{tmp_path}/tmp.mp4',
                            video_save_dir,
                            self.max_clip_len,
                            self.min_clip_len,
                            self.silence_length_crop,
                            self.silence_db_threshold)
            
            print('Video Seperation complete')
            # Seperate clips
            self.extract_audio(title, video_save_dir, audio_save_dir)
            os.remove(path)
            shutil.rmtree('tmp')
            
    def get_intro(self, path):
        clip = VideoFileClip(path)
        # Get first 10% of file (where intro is located)
        clip = clip.subclip(0, int(clip.duration * 0.1))    
        os.makedirs('tmp', exist_ok=True)
        clip.audio.write_audiofile('tmp/tmp1.wav', codec='pcm_s16le')
        clip.close()
        print('Seperating Speakers...')
        output = self.diarize_pipeline('tmp/tmp1.wav')
        intro_clips = []
        print('Locating intro...')
        for idx, (turn, _, speaker) in enumerate(output.itertracks(yield_label=True)):
            if idx == 0: main_speaker = speaker
            if speaker == main_speaker:
                intro_clips.append(turn)
            else:
                break
        print('Found intro, cropping frames...')
        end_time = intro_clips[-2].end
        clip = VideoFileClip(path)
        # Add extra time to end as chunk slicing will require it later
        clip = clip.subclip(0, end_time+0.5)
        clip.audio.write_audiofile('tmp/tmp.wav')
        clip.write_videofile('tmp/tmp.mp4')
        return 'tmp'
            
    def extract_audio(self, title, video_dir, audio_dir):
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        for idx, clip in enumerate(natsorted(os.listdir(video_dir))):
            video = VideoFileClip(f'{video_dir}/{clip}')
            video.set_fps(self.video_fps)
            video.audio.write_audiofile(f'{audio_dir}/{clip.split(".")[0]}.wav')
            
    def save_metadata(self, audio_lens, video_lens, path):
        metadata = {
            'audio_lens':audio_lens,
            'video_lens':video_lens,
        }
        with open(path, 'w') as f:
            f.write((json.dumps(metadata)))

    def start_download(self, link):
        print('Download started')
        yt = YouTube(link)
        new_title = yt.title.split(':')[0].replace(' ', '_').lower()
        raw_path = self.save_path
        new_path = f'{raw_path}/{new_title}.mp4'
        # Check if multiple videos exist
        if os.path.exists(new_path):
            new_idx = 0
            for path in os.listdir(raw_path):
                if new_title in path: new_idx += 1
            new_path = f'{raw_path}/{new_title}_{new_idx}.mp4'
            new_title = f'{new_title}_{new_idx}'
        if os.path.exists(f'{raw_path}/download_log.txt') == False:
            open(f'{raw_path}/download_log.txt', 'a').close()
        with open(f'{raw_path}/download_log.txt', 'r') as f:
            log = f.readlines()
            log = [name.strip('\n') for name in log]
        if new_title not in log: 
            with open(f'{raw_path}/download_log.txt', 'a') as f:
                f.write(f'\n{new_title}')
        yt.streams.get_by_itag(22).download(f'{raw_path}', f'{new_title}.mp4')
        print('Download Complete')
        return new_path, new_title
    
    def crop_frame_size(self, frame):
        H, W, C  = frame.shape
        width_crop = W // 4
        frame = frame[:int(H*0.75), width_crop:-width_crop, :] 
        return frame
    
def remove_short_clips(movie_clips, audio_clips, fps, min_len):
    for i, clip in enumerate(movie_clips):
        print(clip.shape[0])
        if (clip.shape[0]/fps)*1000 <= min_len:
            del movie_clips[i]
            del audio_clips[i]
        
    return movie_clips, audio_clips

def remove_long_clips(movie_clips, audio_clips, fps, max_len):
    for i, clip in enumerate(movie_clips):
        if (clip.shape[0]/fps)*1000 >= max_len:
            del movie_clips[i]
            del audio_clips[i]
        
    return movie_clips, audio_clips


def make_clips_even(movie_clips, audio_clips, fps, sr):
    '''
    Audio clips are list of audio segments 
    can be sliced using ms
    '''
    video_times = [clip.shape[0] / fps for clip in movie_clips]
    audio_times = [audio.duration_seconds for audio in audio_clips]
    for i in range(len(audio_times)):
        # Video longer than audio
        print(video_times[i], audio_times[i])
        if video_times[i] > audio_times[i]:
            crop_to = (video_times[i] * fps) - (audio_times[i] * fps)
            crop_to = movie_clips[i].shape[0] - crop_to
            print('Cropping, video size:', movie_clips[i].shape[0], 'to', crop_to)
            movie_clips[i] = movie_clips[i][:int(crop_to)]
        # Audio longer than video
        elif audio_times[i] > video_times[i]:
            # Convert to ms times and then crop, idxing works in ms for AudioSegment
            crop_to = (audio_times[i] * 1000) - (video_times[i] * 1000)
            crop_to = (audio_times[i] * 1000) - crop_to
            print('Cropping, audio size:', audio_clips[i].duration_seconds * 1000, 'to', crop_to)
            audio_clips[i] = audio_clips[i][:int(crop_to)]
        # assert (audio_clips[i].shape / sr) == movie_clips[i].shape / fps, \
        #     f'Clips are uneven check func. Audio: {audio_clips[i].shape}, Video: {movie_clips[i].shape}'
    return movie_clips, audio_clips
            
            
def equalize_folder(src_dir, target_dir):
    '''Removes wavs of folder that are unequal'''
    src_files = natsorted([file for file in os.listdir(src_dir)])
    tgt_files = natsorted([file for file in os.listdir(target_dir)])[len(src_files):]
    check_files = [file.split('/')[-1].split('.0') for file in src_files]
    for tgt_f in tgt_files:
        if tgt_f in check_files:
            raise ValueError(tgt_f, 'in', check_files)
        os.remove(f'{target_dir}/{tgt_f}')        
        

def get_dataset_length(input_dir):
    times = []
    for folder in tqdm(os.listdir(input_dir)):
        for file in os.listdir(f'{input_dir}/{folder}'):
            audio = AudioSegment.from_wav(f'{input_dir}/{folder}/{file}')
            times.append(audio.duration_seconds)
    print('Total time:', sum(times))
        
