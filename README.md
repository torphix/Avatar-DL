# Work in progress
ASR -> Transformer -> TTS -> Avatar
Updates coming soon

### Avatar
- https://link.springer.com/article/10.1007/s11263-019-01251-8
- Inputs: Audio & single still image 
- Outputs: Talking head video
    ##### TODO:
        - Have img encoder work with any size image
        - Input text instead of audio to speed up inferrence
        - Make video discriminator linear layer variable input
    - Ideas if not working:
        - Video discriminator is not working well: try CNN GRU