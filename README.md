# Summary
Creation of digital avatars that you can conversate with consisting of:
    - Generative adversarial networks for video generation (custom built)
    - ASR system (Finetuned)
    - TTS system (Finetuned)
    - Transformer Text Engine (Finetuned)

Future directions:
    - Upload an ID, Speech clip & Train on custom text data

## Avatar
### Realistic
Uses generative adversarial networks trained on short clips of target speaking
Uses 3 discriminators to enforce generation of realistic images

### Animated
Uses the finetuned style gan modified to work with videos

### Transformer Text Engine 
Uses a finetuned transformer as I will never get as good results training from scratch

## Data pipeline
1. Scrape video and audio pairs from youtube / other sources
2. Train Language model on the Auto generated transcripts
3. Use ASR to transcribe the dataset
4. Finetune the TTS model
    - Normalize wav
    - .wav .lab format
    - Get alignments
5. Generate audio from text to train the generator (after pretraining on real audio)
6. Build end to end streaming pipeline


## Audio Classifier
1. To seperate out the speakers in audio I trained an audio classifer
2. Bubbled up the incorrectly classified wav files and listened to them
3. Found that the model was struggling when there was large amounts of silence at the start (I was cropping the wav files so mainly silence)
4. Add trim silence to data preprocessing pipeline

## TTS
1. Finetune mel decoder with generated spectorgrams for better results

<!-- Goal Get working pipeline ASAP -->
1. Use pretrained ASR
2. Fine tune transformer on lex speech (scrape from youtube audio clips and seperate out lex audio from non lex)
3. Fine tune tts speech model
4. Convert audio in video audio pairs to just work with generated audio
5. Train SDA model on synthetic audio and video
6. Build streaming pipeline end to end
7. Animate?

## App
1. Uses svelte on frontend and flask for backend
2. Two approaches:  
    - Simple: feed audio chunks to ptl module getting video on outside
    - Complex: seperate docker containers running each model (lower latency as each container can run independantly)
3. Current Architecture:
    - Stream audio save to buffer
    - Read 1 sec approx (4-5 syllables per second, avg 2-3 per word)
    - On silence of 1.5 seconds read read entire buffer and clear
    - Pass through ASR model live and return text

<!-- Add Tensorboard to SDA -->

Once done email gdb@openai.com 