import os
import cv2
import dlib
import shutil
import imutils
from tqdm import tqdm
from natsort import natsorted
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from moviepy.video.io import ImageSequenceClip


def align_faces(input_dir, output_dir):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("avatar/realistic/data/face_alignment/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=192, desiredFaceHeight=256)

    # loop over the face detections
    for input_file in tqdm(os.listdir(input_dir)):
        full_input_path = f'{input_dir}/{input_file}'
        print('Processing file: ', input_file)
        v_cap = cv2.VideoCapture(full_input_path)
        fps = v_cap.get(cv2.CAP_PROP_FPS)
        print('FPS:', fps)
        os.makedirs(f'{output_dir}/{input_file.split(".")[0]}', exist_ok=True, )
        idx = 0
        success, image = v_cap.read()
        while success:
            idx += 1
            success, image = v_cap.read()
            if image is None:
                break
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # show the original input image and detect faces in the grayscale
            rects = detector(gray, 2)
            for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                (x, y, w, h) = rect_to_bb(rect)
                faceOrig = imutils.resize(image[y:y + h, x:x + w], width=128)
                faceAligned = fa.align(image, gray, rect)
                # Save images
                cv2.imwrite(f'{output_dir}/{input_file.split(".")[0]}/{idx}_{input_file.split(".")[0]}.png', faceAligned)
        # Convert to movie
        imgs = [img for img in natsorted(os.listdir(f'{output_dir}/{input_file.split(".")[0]}'))]
        clip = ImageSequenceClip.ImageSequenceClip(imgs, fps=fps)
        clip.write_videofile(f'{output_dir}/{input_file.split(".")[0]}.mp4')
        shutil.rmtree(f'{output_dir}/{input_file.split(".")[0]}')
        
    v_cap.release()
    cv2.destroyAllWindows()
        
    