from PIL import Image
import streamlit as st
import cv2
#import urllib.request
import numpy as np

st.title("Image Classification of Apples and Tomatoes")
st.text("Upload a clear image of an apple or a tomato :>")
from image_classification import machine_classification 
uploaded_file = st.file_uploader("Enter image", type=["png","jpeg","jpg"])


if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = machine_classification(image, 'best_models.keras')
    if label>=0 and label<=5:
        st.write("The uploaded item is fresh")
    elif label>=6 and label<=11:
        st.write("The uploaded item is rotten")
    else:
        st.write("improper image or no image uploaded")
