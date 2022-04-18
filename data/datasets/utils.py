import os
from tqdm import tqdm
import subprocess,sys,shlex,re
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from pydub.silence import detect_nonsilent


def split_video_cmd(f_name, 
                    input_file,
                    output_file,
                    max_len,
                    min_len,
                    silence_threshold,
                    min_silence_dur):
    os.makedirs(output_file, exist_ok=True)
    print("Processing: "+input_file)
    findsilence = "ffmpeg -i \""+input_file+"\" -filter_complex \"[0:a]silencedetect=n="+str(silence_threshold)+"dB:d="+ str(min_silence_dur) +"[outa]\" -map [outa] -f s16le -y /dev/null"
    process = subprocess.Popen(shlex.split(findsilence), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,universal_newlines=True)
    (output, err) = process.communicate()
    sections = re.findall('silence_end: ([0-9\.]+).*?silence_start: ([\-0-9\.]+)',output,re.DOTALL|re.MULTILINE)
    print(sections)
    count = 1
    for section in tqdm(sections):
        section_len = float(section[1]) - float(section[0])
        if min_len <= section_len <= max_len:
            if os.path.exists(f'{output_file}/{count}_{f_name}.mp4'):
                continue
            else:
                breakparts = "ffmpeg -ss "+str(float(section[0]))+" -t "+str(float(section[1])-float(section[0]))+" -i \""+input_file+"\" -strict -2 \""+output_file + "/"+str(count)+"_"+f_name+".mp4\""
                breakprocess = subprocess.Popen(shlex.split(breakparts))
                (output, err) = breakprocess.communicate()
                count += 1
        
        
        
def split_video_py(fname,
                   tmp_wav_file,
                   tmp_video_file,
                   output_dir,
                   max_clip_len,
                   min_clip_len,
                   min_silence_len,
                   silence_db):
    '''
    Get time stamps of silences from audio file
    Split video based on the time stamps
    '''    
    os.makedirs(output_dir, exist_ok=True)
    
    # adjust target amplitude
    def _match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    # Convert wav to audio_segment
    audio_segment = AudioSegment.from_wav(tmp_wav_file)

    # Print detected non-silent chunks, which in our case would be spoken words.
    nonsilent_data = detect_nonsilent(audio_segment,
                                      min_silence_len=min_silence_len,
                                      silence_thresh=silence_db,
                                      seek_step=1)
    #convert ms to seconds
    video = VideoFileClip(tmp_video_file)
    clips, filtered_clips = [], []
    for i, chunk in enumerate(nonsilent_data):
        # Video shorter than chunk
        if video.duration < (chunk[1]+250)/1000:
            continue
        else:
            clips.append(video.subclip((chunk[0]-250)/1000, (chunk[1]+250)/1000))
    filtered_clips = []
    for clip in clips:
        if min_clip_len < clip.duration < max_clip_len:
            filtered_clips.append(clip)
    [clip.write_videofile(f'{output_dir}/{i}_{fname}.mp4') for i, clip in enumerate(filtered_clips)]
        