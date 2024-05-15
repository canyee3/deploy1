
from PIL import Image
import streamlit as st
import cv2
#import urllib.request
import numpy as np

model=tf.keras.models.load_model('best_model.keras')

st.title("Image Classification of Apples or Tomatoes")
st.text("Upload a clear image of an apple or a tomato :>")
from image_classification import machine_classification 
uploaded_file = st.file_uploader("Enter image", type=["png","jpeg","jpg"])

size = (300,300)  
if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    image = ImageOps.fit(image, size)
    image = np.asarray(image, dtype = 'float32')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = img / 255
    #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
    #img_reshape = img[np.newaxis,...]
    prediction = model.predict(img)
