'''
This module coordinates the various other models orchestrating
the inferences and reading / writing the buffers required for smooth operation
How it works:
- Spawns a live asr subprocess that streams audio, on silence detected sets 
  silence_detected_event and awaits for silence_detected_event to be unset
- Once silence_detected_event is set Avatar module reads all text from buffer
  and posts in downstream to be processed by language model, reseting the 
  silence_detected_event.
'''
import io
import time
import multiprocessing
from .asr import LiveASR
from queue import  Queue
from .tts.tts import TTS
from .text_engine import TextEngine
from multiprocessing import Process


asr_kwargs = {
    'model_name': "facebook/wav2vec2-base-960h",
    'model_path': '/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/app/backend/models/wav2vec2-base-960h.quant.onnx',
}

class Avatar():
    exit_event = multiprocessing.Event()
    def __init__(self,
                 tts_kwargs,
                 asr_kwargs,
                 text_engine_kwargs,
                 video_engine_kwargs):
        
        self.asr_kwargs = asr_kwargs
        self.tts_kwargs = tts_kwargs
        self.text_engine_kwargs = text_engine_kwargs
        self.video_engine_kwargs = video_engine_kwargs
        
        self.asr_queue = Queue()
        self.text_queue = Queue()
        self.output_queue = Queue()
        
        self.text_engine = TextEngine(**text_engine_kwargs)
        self.tts = TTS(**tts_kwargs)
        
    def start(self):
        self.asr = Process(
            target=Avatar.asr_process,
            args=(self, self.asr_queue, self.text_queue),
            kwargs=self.asr_kwargs)
        self.asr.start()
        
    def write_audio(self, audio):
        self.asr_queue.put(audio)
        
    def asr_process(self, asr_queue, text_queue, **asr_kwargs):
        asr = LiveASR(asr_kwargs)
        asr.start()
        
        while True:
            if Avatar.exit_event.is_set():
                break
            elif asr.silence_detected_event.is_set():
                text = " ".join([asr.asr_output_queue.get() 
                                 for _ in range(asr.asr_output_queue.qsize())])
                text_queue.put(text)
                # If you want to continue transcribing spawn compute response in a different process
                text, audio = self.compute_response(text_queue)      
                self.output_queue.put([text, audio]) # Add video to queue as well
                # self.create_video()
                asr.silence_detected_event.clear()
                continue
            else:
                audio_frames = asr_queue.get()
                asr.write_to_buffer(audio_frames)
            
    def compute_response(self, text_queue):
        text = " ".join([text for text in text_queue.get()])
        response = self.text_engine(text)
        audio = self.tts(response)
        # video = self.sda(audio)
        return response, audio, # video
        
    def create_video(self, text, audio, video):
        pass
        
        
            