# Summary
Creation of digital avatars that you can conversate with consisting of:
   \n - Generative adversarial networks for video generation (custom built)
   \n - ASR system (Finetuned)
   \n - TTS system (Finetuned)
   \n - Transformer Text Engine (Finetuned)

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
