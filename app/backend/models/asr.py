'''
Live streaming ASR system
See if multiprocessing leads to speed increases
'''
import os
import io
import sys
import time
import torch
import pyaudio
import librosa
import argparse
import threading
import webrtcvad
import numpy as np
from queue import  Queue
import onnxruntime as rt
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2VecOnnxInference():
    def __init__(self, processor_name, onnx_path):
        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)        
        options = rt.SessionOptions()
        options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(onnx_path, options)
        
    def audio_buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ''
        
        inputs = self.processor(torch.tensor(audio_buffer), 
                                sampling_rate=16000,
                                return_tensors='np',
                                padding=True)    

        output = self.model.run(None, {self.model.get_inputs()[0].name: inputs.input_values})[0]
        predictions = np.argmax(output, axis=-1)
        text = self.processor.decode(predictions.squeeze().tolist())
        return text
    
    def file_inference(self, filepath):
        audio_input, samplerate = librosa.load(filepath, sr=16000)
        return self.audio_buffer_to_text(audio_input)
  
  
class LiveASR():
    exit_event = threading.Event()
    silence_detected_event = threading.Event()
    def __init__(self, model_name, model_path, silence_threshold=1.5):
        '''
        param: silence_threshold: in seconds defaults to 1.5 seconds
        param: model_name: used to load the asr text processor
        Audio should be read and written to audio_buffer
        Once silence threshold is crossed data is shipped downstream
        '''
        self.model_name = model_name
        self.model_path = model_path
        # Audio settings
        self.audio_buffer = io.BytesIO(b"")
        self.silence_threshold = silence_threshold
        
    def start(self):
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()
        
        self.asr_process = threading.Thread(
            target=LiveASR.asr_process, 
            args=(self.asr_input_queue, 
                  self.asr_output_queue,
                  self.audio_buffer))     
        self.asr_process.start()
        time.sleep(5) # ASR process must be initialized before voice detector
        
        self.vd_process = threading.Thread(
            target=LiveASR.vd_process,
            args=(self.asr_input_queue,
                  self.audio_buffer,
                  self.silence_threshold))
        self.vd_process.start()
        
    def stop(self):
        LiveASR.exit_event.set()
        self.asr_input_queue.put("close")
        print('ASR session terminated')
        
    def write_to_buffer(self, audio_bytes):
        self.audio_buffer.write(audio_bytes)
        return
        
    def vd_process(asr_input_queue,
                   audio_buffer,
                   silence_threshold):
        '''
        Detects if voice is in segment
        / silence threshold reached
        '''
        vad = webrtcvad.Vad()
        vad.set_mode(1)

        SAMPLE_RATE = 16000
        FRAME_DURATION = 30 # ms 10, 20 or 30 for voice detection
        CHUNK_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

        speech_frames = b''
        silence_frames = b''
        while True:
            if LiveASR.exit_event.is_set():
                break
            elif len(silence_frames) > int(silence_threshold*SAMPLE_RATE):
                # Send signal to pass transcription downstream
                if len(speech_frames) > 1:
                    asr_input_queue.put(speech_frames)
                    # TODO process transcription
                silence_frames = b''
                speech_frames = b''
                LiveASR.silence_detected_event.set()
            elif LiveASR.silence_detected_event.is_set():
                time.sleep(2)
                continue
            
            frame = audio_buffer.read(CHUNK_SIZE)
            is_speech = vad.is_speech(frame, SAMPLE_RATE)            
            if is_speech:
                # Reset silence buffer & add to speech_buffer
                silence_frames = b''
                speech_frames += frame
            else:
                # Add frame to silence (for detector)
                # Add & clear current speech buffer frames
                if len(speech_frames) > 1:
                    asr_input_queue.put(speech_frames)
                silence_frames += frame
                speech_frames = b''
                
    def asr_process(model_name, model_path, in_queue, output_queue):
        if model_path.endswith('onnx'):
            assert os.path.exists(model_path), \
                f'Onnx model path not found, {model_path}'
            wav2vec_asr = Wav2VecOnnxInference(model_name, model_path)            

        else:
            # TODO normal pth inference
            pass
        
        print('Transcription Started')
        while True:
            audio_frames = in_queue.get()
            if audio_frames == 'close':
                break
            float_buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767
            text = wav2vec_asr.audio_buffer_to_text(float_buffer).lower()
            if text != '':
                output_queue.put(text)
            elif 'goodybe' in text:
                break
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-wp', '--wav_path', required=True)
    args, leftover_args = parser.parse_known_args()
    asr = Wav2VecOnnxInference("facebook/wav2vec2-base-960h",
                               "wav2vec2-base-960h.quant.onnx")
    text = asr.file_inference(args.wav_path)
    print(text)
    