import streamlit as st
# import moviepy.editor as moviepy
from hacktest import hack1
from hume2 import main3
# from choice5 import choice5
import os
from PIL import Image
import numpy as np


def main():
    new_title = 'SECURE LIVE VIDEO FEED FOR DEEPFAKE PREVENTION'
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # image = Image.open(os.path.join(base_dir, "logo.png"))
    # image = np.asarray(image)
    # st.image(image, width=100)
    read_me_0 = st.markdown(new_title, unsafe_allow_html=False)
    read_me = st.markdown("""INFORMATION\n
    Created by Dodam (Amelie) Yoon\n
    This is a prototype for a secured live video application with real time 
    facial recognition and deepfake detection. The prototype runs on XRP Ledger 
    technology, which secures each "transaction" (ie. video feed) with a hash 
    value that is watermarked into the video feed using stegonagraphic Least 
    Significant Bit embedding. The web prototype is created with streamlit with 
    Python, with a ResNet50 backboned ML model. The facial recognition and 
    landmark detection is done through the dlib library, with similarity 
    transformation between two matrices using the Kabsch-Umeyama algorithm. 
    Avatarify is used as a test case against this model. 
    
    \nTHE PROBLEM\n
    Deepfakes are already a massive problem, and will become a billions of
    dollars market in the upcoming years. By using live deepfake video calls 
    (a technology which is already becoming an issue), a bad actor could 
    potentially con someone out of finances by pretending to be someone else, 
    blackmail others by acting as them in compromising positions, and sabotage 
    the lives of many. 
    
    Attempts at protecting against generative AI has been a constant focus for 
    R&D. Teams from MIT and the University of Chicago created adversarial attack 
    networks that would create perturbations in data so that one's media could 
    be protected, however, it was easily circumvented by compressing images to JPEG, 
    applying bilateral filters, as well as Fourier Transform for signal denoising.
    
    \nTHE SOLUTION\n 
    By securing live video feed powered by blockchain technology, one would thwart 
    attackers, not by defeating their attacks, but proving it too computationally 
    consuming for them to even bother and attempt to spoof data. 
    
    Rule based systems are imposed. A hash value would be embedded into the video 
    feed, taken from XRP Ledger technology, preventing a bad actor from potentially 
    video calling from another digital screen where they are able to circumvent 
    the security protocols of the live video software (for more information, please 
    look at document liveness detection).
    
    \nUSE CASES\n 
    Say someone is accepting a job offer remotely. Both employer and employee use 
    the application to securely video call and confirm the offer before a financial 
    transaction takes place.
    
    \nMARKETABILITY\n 
    This product can be sold as its own "DeepFake-protected Zoom" software, or sold as 
    a SaaS to other video call companies to integrate into their software. Currently, 
    the market for fraud detection and identity verification is $30 billion, with a 
    growing market for deepfake detection and prevention.
    
    github: https://github.com/strudelmix/ )""")

    st.sidebar.title("Select Activity")
    c1 = "1: Embedded Live Detector/Protection"
    c2 = "2: Bounding Boxes"
    c3 = "3: Steganography Hash Value"
    choice = st.sidebar.selectbox("MODE", ("About", c1))
    # no.1 = mine, wavelet, no.2 = mine, iphone, no.3 = romain's
    if choice == c1:
        read_me_0.empty()
        read_me.empty()
        hack1()
    # elif choice == c3:
    #    read_me_0.empty()
    #    read_me.empty()
    #    main1(steg=True)




    elif choice == "About":
        print()


if __name__ == '__main__':
    main()
