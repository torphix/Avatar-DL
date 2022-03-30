# Work in progress
ASR -> Transformer -> TTS -> Avatar
Updates coming soon

# Overview
- A series of neural networks that creates an AI companion
- Motivation:
    - I wanted to create an end 2 end deep learning product that can be used by consumers
    - I wanted to implement a large range of different architectures and data pipelines
    - I wanted to harvest and process my own data not just use prebuilt cleaned datasets
    - Architectures used:
        - Generative Adversarial data for 3D data (Realistic Avatar)
        - Pytorch 3D ie: deep learning on meshs (Animated Avatar)
        - Transformers: Text engine
        - Automatic Speech Recognition
        - Text to Speech engine 

### Avatar
#### Real
#### Animated
    - Avatar head
    - Wav file -> NN -> Series of facial anchor points
    - Training:
        - Dataset:
            - Convert videos of target into mesh points frame by frame
            - Use the locations as targets for each wav frame similar to the generator used in real avatar
    - Starting anchor points of 2d image and 3d mesh
    - At each frame calculate the transformation that occurs between the 2d points and then apply to 3d


### TTS
#### Data Pipeline
- Speaker diarization:
    - Hand labelled 50 samples of audio snippets from conversations
    - Train audio classifier to distinguish clips 
    - Train ASR model to predict text from speech


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
