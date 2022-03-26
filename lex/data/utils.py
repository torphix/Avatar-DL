from moviepy.editor import VideoFileClip

    
    
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
    remove_idxs = []
    for i, clip in enumerate(movie_clips):
        if (clip.shape[0]/fps)*1000 <= min_len:
            remove_idxs.append(i)
    print(remove_idxs)
    for idx in remove_idxs:
        del movie_clips[idx]
        del audio_clips[idx]
        
    return movie_clips, audio_clips


def remove_uneven_clips(movie_clips, audio_clips, fps, sr, threshold):
    video_times = [clip.shape[0] / fps for clip in movie_clips]
    audio_times = [audio.duration_seconds for audio in audio_clips]
    print(video_times, audio_times)
    for i in range(len(audio_times)):
        print(video_times[i], audio_times[i])
        if abs(video_times[i] - audio_times[i]) > threshold:
            del movie_clips[i]
            del audio_clips[i]
    return movie_clips, audio_clips
            