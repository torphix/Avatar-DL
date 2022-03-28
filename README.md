# Work in progress
ASR -> Transformer -> TTS -> Avatar
Updates coming soon

### Avatar

Animated Unity 3D pack animation kit for dataset creation


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

## Data Processing Pipeline
    - Training Realistic Avatar
        - 1. Faces aligned with the eyes and mouth in approximatly same coordinates
        - 2. 
### Face alignment
    - cv2 used to align faces in frame
    - Faces center cropped removing as much border as possible


<!-- Next -->
1. Lex:
    - Center crop and align face what is average face alignment?
    - Reconstruction should happend exclusivly around the mouth / facial area
    
    - Videos Done:
        - Ian Goodfellow
