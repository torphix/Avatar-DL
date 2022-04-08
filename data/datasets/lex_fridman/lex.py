import os
from natsort import natsorted
import numpy as np
from tqdm import tqdm
from pytube import YouTube
from pydub import AudioSegment, silence
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip, ImageSequenceClip


def create_lex_dataset(links,
                       silence_length_crop,
                       video_fps,
                       max_frames,
                       min_frame_len,
                       max_frame_len,
                       save_path):
    data_handler = LexDataHandler(
                        links,
                        silence_length_crop,
                        video_fps,
                        max_frames,
                        min_frame_len,
                        max_frame_len,
                        save_path)
    data_handler.start()
    # align_dataset([f'{root_input}/{file}' for file in os.listdir(root_input)])    
    

class LexDataHandler(object):
    def __init__(self,
                 links, 
                 silence_length_crop,
                 video_fps,
                 max_frames,
                 min_frame_len,
                 max_frame_len,
                 save_path,
                 ) -> None:
        '''
        links: Youtube links
        silence_length_crop: in milli seconds
        max_frames: if memory contraint is a problem -1 for no limit
        max_frame_len: removes clips & audio if len is >= in ms
        min_frame_len: removes clips & audio if len is <= in ms
        '''
        self.video_fps = video_fps
        self.links = links
        self.silence_length_crop = silence_length_crop
        self.max_frames = max_frames
        self.min_frame_len = min_frame_len
        self.max_frame_len = max_frame_len
        self.save_path = save_path

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
        lex_processed_data_path = self.save_path
        os.makedirs(f'{abs_path}/{lex_processed_data_path}/VideoFlash/{f_name}', exist_ok=True)
        os.makedirs(f'{abs_path}/{lex_processed_data_path}/AudioWAV/{f_name}', exist_ok=True)
        video_root = f'{abs_path}/{lex_processed_data_path}/VideoFlash/{f_name}'
        audio_root = f'{abs_path}/{lex_processed_data_path}/AudioWAV/{f_name}'
        clip = ImageSequenceClip([frame for frame in clip], fps=fps)
        clip.write_videofile(f'{video_root}/{idx}_{f_name}.mp4', fps)
        audio.export(f'{audio_root}/{idx}_{f_name}.wav', format='wav')
        print('Save complete')
        
    def start_download(self, link):
        print('Download started')
        yt = YouTube(link)
        new_title = yt.title.split(':')[0].replace(' ', '_').lower()
        abs_path = os.path.abspath('.')
        raw_path = self.save_path
        new_path = f'{abs_path}/{raw_path}/{new_title}.mp4'
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
        movie_clips, audio_clips = remove_long_clips(movie_clips, audio_clips, fps, self.max_frame_len)
        # Crop uneven clips (do this in dataset)
        movie_clips, audio_clips = make_clips_even(movie_clips, audio_clips, fps, clip.audio.fps)
        movie_clips, audio_clips = make_clips_even(movie_clips, audio_clips, fps, clip.audio.fps)
        assert len(audio_clips) == len(movie_clips),\
            f'Audio clips {len(audio_clips)}, Movie clips {len(movie_clips)} unequal'
        return movie_clips, audio_clips, fps
        
        
    
    
def seperate_video(file_name, input_path, output_path):
    video = VideoFileClip(f'{input_path}/{file_name}')
    video.write_videofile(f'{output_path}/VideoFlash/{file_name}.mp4', audio=False)
    video.audio.write_audiofile(f'{output_path}/AudioWAV/{file_name}.wav')
    

def get_cut_idxs_for_video(clips_lens):
    running_idx = 0
    movie_clip_cut_idxs = []
    for mcl in clips_lens:
        running_idx += mcl
        movie_clip_cut_idxs.append(running_idx)
    return movie_clip_cut_idxs

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
        
        
# src_root = '/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/data/datasets/lex_fridman/raw/VideoFlash'
# tgt_root = '/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/data/datasets/lex_fridman/raw/AudioWAV'
# src = natsorted([folder for folder in os.listdir(src_root)])
# tgts = natsorted([folder for folder in os.listdir(tgt_root)])

# for i in range(len(src)):
#     equalize_folder(f'{src_root}/{src[i]}', f'{tgt_root}/{tgts[i]}')
    