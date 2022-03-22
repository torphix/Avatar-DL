# Work in progress
ASR -> Transformer -> TTS -> Avatar
Updates coming soon

### Avatar
## - Status: Videos are generating, Requires ID frame and audio
- https://link.springer.com/article/10.1007/s11263-019-01251-8
- Inputs: Audio & single still image 
- Outputs: Talking head video
    ##### TODO:
        - Have img encoder work with any size image
        - Input text instead of audio to speed up inferrence
        - Make video discriminator linear layer variable input
        - Adjust dims for batch processing
        - Add an emotion embedding for each video
