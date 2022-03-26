import re
import os
import numpy as np
from tqdm import tqdm
from pytube import YouTube
from pydub import AudioSegment, silence
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip, ImageSequenceClip
from utils import get_cut_idxs_for_video, remove_short_clips, remove_uneven_clips

class LexDataHandler(object):

    def __init__(self,
                 links, 
                 silence_length_crop,
                 video_fps,
                 max_frames=1500,
                 min_frame_len=1000) -> None:
        '''
        links: Youtube links
        silence_length_crop: in milli seconds
        max_frames: if memory contraint is a problem -1 for no limit
        min_frame_len: removes clips & audio if len is <= in ms
        '''
        self.video_fps = video_fps
        self.links = links
        self.silence_length_crop = silence_length_crop
        self.max_frames = max_frames
        self.min_frame_len = min_frame_len

    def start(self):
        print('Starting youtube download')
        for link in tqdm(self.links):
            path, title = self.start_download(link)
            movie_clips, audio_clips, fps = self.process_video(path)
            # Save data
            for idx, clip in enumerate(movie_clips):
                clip = np.stack(clip)
                self.save_audio_and_video(idx, title, clip,
                                          audio_clips[idx], fps)
                
    def save_audio_and_video(self, idx, f_name, clip, audio, fps):
        print('Saving video')
        abs_path = os.path.abspath('.')
        os.makedirs(f'{abs_path}/lex/data/processed/VideoFlash/{f_name}', exist_ok=True)
        os.makedirs(f'{abs_path}/lex/data/processed/AudioWAV/{f_name}', exist_ok=True)
        video_root = f'{abs_path}/lex/data/processed/VideoFlash/{f_name}/'
        audio_root = f'{abs_path}/lex/data/processed/AudioWAV/{f_name}/'
        clip = ImageSequenceClip([frame for frame in clip], fps=fps)
        clip.write_videofile(f'{video_root}/{idx}_{f_name}.mp4', fps)
        audio.export(f'{audio_root}/{idx}_{f_name}.wav', format='wav')
        print('Save complete')
        
    def start_download(self, link):
        print('Download started')
        yt = YouTube(link)
        new_title = yt.title.split(':')[0].replace(' ', '_').lower()
        abs_path = os.path.abspath('.')
        new_path = f'{abs_path}/lex/data/raw/{new_title}.mp4'
        with open('lex/data/download_log.txt', 'r') as f:
            log = f.readlines()
            log = [name.strip('\n') for name in log]
        if new_title in log: return new_path, new_title
        else: 
            with open('lex/data/download_log.txt', 'a') as f:
                f.write(f'\n{new_title}')
                
        yt.streams.get_by_itag(22).download('lex/data/raw', f'{new_title}.mp4')

        print('Download Complete')
        return new_path, new_title
    
    def crop_frame_size(self, frame):
        H, W, C  = frame.shape
        width_crop = W // 4
        frame = frame[:int(H*0.75), width_crop:-width_crop, :] 
        return frame
        
    def process_video(self, path):
        '''
        Read video & audio
        Get intro by fade out frame
        Slice by silence 
        Return sliced videos & audios
        '''
        print('Processing started')
        clip = VideoFileClip(path)
        clip = clip.set_fps(self.video_fps)
        fps = clip.fps
        print('FPS Is:', fps)
        frames = clip.iter_frames()
        intro_frames = []
        for i, frame in enumerate(frames):
            # Trade off between speed and memory consumption
            frame = self.crop_frame_size(frame)
            if np.all(frame == 0) or len(intro_frames) == self.max_frames:
                if i == self.max_frames: print('Max frames reached')
                intro_frames = np.stack(intro_frames)
                break
            else:
                intro_frames.append(frame)
                
        print('Cropping frames by silence')
        # Crop frames by silence
        frame_len = intro_frames.shape[0]
        audio_len = int((frame_len / fps) * clip.audio.fps)
        audio = clip.audio.to_soundarray(
            fps=clip.audio.fps,
            nbytes=2)[:audio_len]
        audio = AudioArrayClip(audio, fps=clip.audio.fps)
        tmp_save_loc = 'lex/tmp/tmp_lex_data.wav'
        os.makedirs('lex/tmp', exist_ok=True)
        audio.write_audiofile(tmp_save_loc, 
                              fps=clip.audio.fps)
        # Audio becomes corrputed when directly loading as bytes
        audio = AudioSegment.from_wav(tmp_save_loc)
        os.remove(tmp_save_loc)
        audio = audio.set_channels(1)
        audio_clips = silence.split_on_silence(audio,
                                               min_silence_len=self.silence_length_crop,
                                               silence_thresh=-40,
                                               keep_silence=True,
                                               seek_step=int(1000 / fps)) # should be equivalent to 1 video frame)
        # Crop video based on audio lengths
        audio_clip_lens = [audio_clip.duration_seconds
                           for audio_clip in audio_clips]
        movie_clip_lens = [int(audio_clip_len * fps)
                           for audio_clip_len in audio_clip_lens]
        movie_clip_cut_idxs = get_cut_idxs_for_video(movie_clip_lens)
        movie_clips = np.split(intro_frames, movie_clip_cut_idxs)[:-1]
        movie_clips, audio_clips = remove_short_clips(movie_clips, audio_clips, fps, self.min_frame_len)
        movie_clips, audio_clips = remove_uneven_clips(movie_clips, audio_clips, fps, clip.audio.fps, threshold=0.1)
        assert len(audio_clips) == len(movie_clips),\
            f'Audio clips {len(audio_clips)}, Movie clips {len(movie_clips)} unequal'
            
        return movie_clips, audio_clips, fps
        
        
LINKS = [
    'https://www.youtube.com/watch?v=ez773teNFYA',
    # 'https://www.youtube.com/watch?v=LDTe8uFqbws',
    # 'https://www.youtube.com/watch?v=EYIKy_FM9x0',
    # 'https://www.youtube.com/watch?v=uPUEq8d73JI',
    # 'https://www.youtube.com/watch?v=E1AxVXt2Gv4',
    # 'https://www.youtube.com/watch?v=8A-5gIW0-eI',
    # 'https://www.youtube.com/watch?v=13CZPWmke6A',
    # 'https://www.youtube.com/watch?v=U5OD8MjYnOM',
    # 'https://www.youtube.com/watch?v=TRdL6ZzWBS0',
    ]
silence_length_crop = 600
max_frames = 4000
min_frame_len = 1000
video_fps = 25

data_handler = LexDataHandler(LINKS,
                              silence_length_crop,
                              25,
                              max_frames,
                              min_frame_len)
data_handler.start()
# movie_clips, audio_clips, fps = data_handler.process_video('/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/lex/data/raw/david_silver.mp4')
# for idx, clip in enumerate(movie_clips):
    # data_handler.save_audio_and_video(idx,
                                    #   'david_silver',
                                    #   clip,
                                    #   audio_clips[idx],
                                    #   fps)
'''
Test class
1. Download link
2. Find & crop out intro
3. Crop intro by silence frames
4. Reduce height and width of images
5. Save clips and audio
# Next
6. Align to face
7. Create dataset
8. Train
9. 
'''