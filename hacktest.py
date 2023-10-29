import time

import numpy
import streamlit as st
import torch
import cv2, os, dlib
import numpy as np
from CNNDetection.py_utils.face_utils import lib
import torch.nn
import torchvision.transforms as transforms
from PIL import Image
from CNNDetection.networks.resnet import resnet50
from xrpltest import get_account, send_xrp
from stegtest import genData, encode

# this program contains a modified version of https://github.com/peterwang512/CNNDetection from the paper
# Detecting CNN-Generated Images [[Project Page]] [Sheng-Yu Wang](https://peterwang512.github.io/), [Oliver Wang]
# (http://www.oliverwang.info/), [Richard Zhang](https://richzhang.github.io/), [Andrew Owens]
# (http://andrewowens.com/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/).<br>In [CVPR]
# (https://arxiv.org/abs/1912.11035), 2020.
# References
# [1] Wang, S.Y., Wang, O., Zhang, R., Owens, A., & Efros, A. (2020). CNN-generated images are surprisingly easy to spot...for now. In CVPR.
# [2] Avatarify, https://github.com/alievk/avatarify

# License for umeyama transform is provided in umeyama.py
# LICENSE.txt provides further information


# INFORMATION
# This is a prototype for a secured live video application with real time facial recognition and deepfake detection.
# The prototype runs on XRP Ledger technology, which secures each transaction (ie. video feed) with a hash value
# that is watermarked into the video feed using stegonagraphic Least Significant Bit embedding
# The web prototype is created with streamlit with Python, with a ResNet50 backboned ML model.
# The facial recognition and landmark detection is done through the dlib library, with similarity transformation
# between two matrices using the Kabsch-Umeyama algorithm.
# Avatarify is used as a test case against this model.

# mindsdb 1k + offer
# 2k for convex
# 1k for xrpl
# 2k for intersystems
# ai x crypto $3.5k
# 1k hume + meet n greet ceo
# wispr interview
# 3k axelar
# 1.5k reflex

# humeAI? cognitive/mood detector


base_dir = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(base_dir, 'CNNDetection', 'weights', 'blur_jpg_prob0.5.pth')
dlib_dir = os.path.join(base_dir, 'CNNDetection', 'dlib_model', 'shape_predictor_68_face_landmarks.dat')
use_cpu = False

front_face_detector = dlib.get_frontal_face_detector()
lmark_predictor = dlib.shape_predictor(dlib_dir)

model = resnet50(num_classes=1)
state_dict = torch.load(weights_dir, map_location='cpu')
model.load_state_dict(state_dict['model'])
if not use_cpu:
    model.cuda()
model.eval()

# Transform
trans_init = []
trans = transforms.Compose(trans_init + [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def rotate_face(face_image, p30, p8):
    """Rotates face around the nose. Nose -point 30, podborodok = 8
    """
    v = p8 - p30
    if v[1] == 0:
        return face_image
    else:
        angle = -np.arctan(v[0] / v[1])
        return rotate(face_image, np.rad2deg(angle), tuple(p30))


##
def cut_head(imgs, point):
    #     return imgs
    h, w = imgs.shape[:2]
    x1, y1 = np.min(point, axis=0)
    x2, y2 = np.max(point, axis=0)
    delta_x = (x2 - x1) / 5
    delta_y = (y2 - y1) / 5

    x1_ = np.int(np.maximum(0, x1 - 0.65 * delta_x))
    x2_ = np.int(np.minimum(w - 1, x2 + 0.65 * delta_x))
    y1_ = np.int(np.maximum(0, y1 - 3.5 * delta_y))
    y2_ = np.int(np.minimum(h - 1, y2 + 0.001 * delta_y))

    return imgs[y1_:y2_, x1_:x2_, :]


# takes about 20 seconds to generate both sending and destination wallet. In practice and as a prototype it is fine, but when showing to judges
# I'd rather use static addresses I have already generated for the wallets.
# seed = ''
# sending_wallet = get_account(seed)
# destination_wallet = get_account(seed)

# sending_id = sending_wallet.classic_address
# destination_id = destination_wallet.classic_address


sending_id = 'rBxmj78N2cCYPkKfpP55zJqFDYfaaRqGxk'
destination_id = 'rBJDj5c12jZsm1Yt7UyTYSv6rsxpyrfxhy'



# def placeholder():
#     st.write("Sender address: " + sending_id)
 #   st.write("Destination address: " + destination_id)


def explain_hash(hash_value, run):
    if run:
        if not hash_value:
            st.write("")
        else:
            st.write(
                str(hash_value) + " is the ledger hash value. This hash value is steganographically embedded into the live video feed using Least Significant Bits.")


def main(use_cpu=use_cpu, number_of_samples=1, rotate_image=False, bounding_box=False):
    st.title("Webcam Live Feed")
    FRAME_WINDOW = st.image([])
    # local webcam only
    camera = cv2.VideoCapture(0)

    hash_value = send_xrp()
    run = st.checkbox('Run')
    if run:
        placeholder = st.text("Securing connection between parties...")
        time.sleep(2)
        placeholder.empty()

        if not hash_value:
            st.write("ending connection...")
            return

    explain_hash(hash_value, run)


    with st.empty():
        while run:

            _, image = camera.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_info = lib.align(image, front_face_detector, lmark_predictor)

            # Samples
            if (len(face_info) > 0):
                output_text = str(len(face_info)) + ' faces are detected'
                # st.write(len(face_info), 'faces are detected')

                bbox = []
                for k in range(len(face_info)):  # for loop on number of detected faces
                    _, point = face_info[k]

                    if bounding_box:
                        bbox.append((*np.min(point, axis=0), *np.max(point, axis=0)))

                    # for i in range(number_of_samples):
                    #    if number_of_samples > 1:
                    #        roi, _ = lib.cut_head([image], point, i) if not rotate_image else lib.cut_head(
                    #            [rotate_face(image, point[30], point[8])], point, i)  # Or use cut_head
                    #    else:
                    roi = [cut_head(image, point)] if not rotate_image else [
                        cut_head(rotate_face(image, point[30], point[8]), point)]  # Or use cut_head

                    img_tensor = torch.tensor(0)
                    if roi:
                        current_image = Image.fromarray(roi[0])
                        img_tensor = trans(current_image)

                    # imgs_tensor = [trans(img) for img in samples]

                    # _results = []
                    with torch.no_grad():
                        if roi:
                            # for img in imgs_tensor:
                            in_tens = img_tensor.unsqueeze(0)
                            if not use_cpu: in_tens = in_tens.cuda()
                            # _results.append(model(in_tens).sigmoid().item())
                            _results = model(in_tens).sigmoid().item()
                            st.write(output_text + '\nprobability of being synthetic: ' + str(round(_results, 2)) + '%')
                        # if sample size is more than 1, do below but i dont wanna
                        # also am not appending the probability prediction for every frame into a list, keep it simple
                        # probabilities.append(mean(_results))
                # if bounding_box:
                #    with open('bound_box', 'w') as f:
                #        f.write(str(dict(zip(bbox, probabilities))))
            if (len(face_info) == 0):
                # output_text = 'warning! No face has been found!'
                st.write('warning! No face has been found!')

            # steganography of hash_value here
            # list of binary code for hash value
            encoded_PIL = encode(image, hash_value)
            open_cv_image = numpy.array(encoded_PIL)

            FRAME_WINDOW.image(open_cv_image)

        else:
            st.write('Play video feed')




if __name__ == '__main__':
    main()
