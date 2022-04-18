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

## Data pipeline / Summary of steps
1. Scrape video from youtube 
2. Clean dataset filter out too long / too short etc
3. Seperate speakers by clustering speaker embeddings and selecting for desired cluster
4. Transcribe remaining audio files
5. Manually fix transcriptions
6. Fine tune ASR model on fixed transcriptions and updated lexicon
7. Transcribe remaining files
8. Fine tune TTS model on text audio pairs
9. Generate SDA dataset text-audio (synthetic)-video pairs
10. Align face in center and get mouth landmarks
10. Train SDA model, reconstruction loss should be around mouth landmarks 
11. Fine tune LM on subject text
12. Build end to end streaming inference system

## Lessons learnt
1. Bubble up any misclassified elements and fine tune on errors in dataset

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


441