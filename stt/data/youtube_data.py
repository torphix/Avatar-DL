import os
import textgrid as tg
from tqdm import tqdm
from pytube import YouTube
from natsort import natsorted
from pydub import AudioSegment 


def create_dataset(links, output_dir):
    '''
    Creates audio / text pairs from list of links provided
    '''
    creator = YTTTSDatasetCreator(links, output_dir)
    creator.start()

class YTTTSDatasetCreator(object):
    def __init__(self, links, output_dir, seperate_speakers=False) -> None:
        '''
        :param: links: Links to youtube videos
        '''
        self.links = links
        self.output_dir = output_dir
        
    def start(self):
        print('Starting youtube download')
        for i, link in tqdm(enumerate(self.links)):
            self.start_download(link, i)
                
    def start_download(self, link, vid_idx):
        print(f'Download started, {vid_idx}')
        yt = YouTube(link)
        # new_title = yt.title.split(':')[0].replace(' ', '_').lower()
        new_title = 'lex'
        raw_path = 'data/datasets/raw'
        yt.streams.filter(only_audio=True).first().download(f'{raw_path}', f'{new_title}.mp3')
        caption = yt.captions.get('en',None)
        if caption is None:
            caption = yt.captions.get('a.en',None)
            if caption is None:
                return 
        captions = caption.generate_srt_captions()
        captions = captions.split('\n\n')
        # Split audio
        audio = AudioSegment.from_file(f'{raw_path}/{new_title}.mp3')
        captions, audio = self.process_datapoint(captions, audio)
        self.save_audio_and_video(new_title, captions, audio)

    def process_datapoint(self, captions, audio):
        def _time_to_ms(time):
            milli = int(time.split(',')[1])
            time = time.split(',')[0]
            hours = int(time.split(':')[0])
            mins = int(time.split(':')[1])
            secs = int(time.split(':')[2])
            seconds = (hours * 3600) + (mins * 60) + secs
            ms = (seconds * 1000) + milli
            return ms
        
        audio_chunks = []
        caption_text = []
        for caption in captions:
            time = caption.split('\n')[1]
            start_time = _time_to_ms(time.split('-->')[0])
            end_time = _time_to_ms(time.split('-->')[1])
            audio_chunks.append(audio[int(start_time): int(end_time)])
            caption_text.append(caption.split('\n')[-1].strip('\n'))
        assert len(audio_chunks) == len(captions), \
            'Audio chunks and captions are an unequal length!'
        return captions, audio_chunks
                
    def save_audio_and_video(self, f_name, captions, audio):
        print('Saving files....')
        for i in tqdm(range(len(audio))):
            os.makedirs(f'{self.output_dir}/MFA/{f_name}', exist_ok=True)
            audio_chunk = audio[i]
            caption = captions[i]
            caption = caption.split('\n')[-1]
            with open(f'{self.output_dir}/MFA/{f_name}/{i}_{f_name}.lab', 'w') as f:
                f.write(caption)
            audio_chunk.export(f'{self.output_dir}/MFA/{f_name}/{i}_{f_name}.wav', format='wav')
        
        
# Either caption timiigns are incorrect
# Time roundings are incorrect (in my function)


def crop_audio_to_textgrid(tg_dir, audio_dir):
    audio_files = []
    for file in os.listdir(audio_dir):
        if file.endswith('wav'):
            audio_files.append(file)
    audio_files = natsorted(audio_files)
    for i, tg_file in tqdm(enumerate(natsorted(os.listdir(tg_dir)))):
        assert tg_file.split('.')[0] == audio_files[i].split('.')[0], \
            'Tg file and audio file are not the same'
        tg_file = tg.TextGrid.fromFile(f'{tg_dir}/{tg_file}')
        start_times, end_times = [], []
        for interval in tg_file[0]:
            if interval.mark == '':
                continue
            else:
                start_times.append(interval.minTime)
                end_times.append(interval.maxTime)
        start_time = min(start_times)
        end_time = max(end_times)
        audio = AudioSegment.from_file(f'{audio_dir}/{audio_files[i]}')
        audio = audio[start_time*1000: end_time*1000]
        print(f'{audio_dir}/{audio_files[i]}')
        audio.export(f'{audio_dir}/{audio_files[i]}', format='wav')
