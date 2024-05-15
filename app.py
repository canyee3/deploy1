from PIL import Image
import tensorflow as tf
import streamlit as st
import cv2
#import urllib.request
import numpy as np
import glob
from PIL import Image, ImageOps

@st.cache(allow_output_mutation=True)

def import_and_predict(image_data, model):
        size = (300,300)  
        image = ImageOps.fit(image_data, size)
        image = np.asarray(image, dtype = 'float32')
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = img / 255
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        #img_reshape = img[np.newaxis,...]
        prediction = model.predict(img)
        return prediction

model = tf.keras.models.load_model('final_model.hdf5')
st.title("Image Classification of Apples and Tomatoes")
st.text("Upload a clear image of an apple or a tomato :>")
uploaded_file = st.file_uploader("Enter image", type=["png","jpeg","jpg"])

if uploaded_file is None:
    st.text("Please upload an image file")
        
else:
    size = (300,300) 
    image = Image.open(uploaded_file)
    image = ImageOps.fit(image, size)
    st.image(image, width = image.size[0]*2)
    prediction = import_and_predict(image, model)
    #prediction = model.predict(image)
    score = tf.nn.softmax(prediction[0])
    #st.write(prediction)
    #st.write(score)
    string = "This image most likely a {} with a {:.2f}% confidence.".format(columns[np.argmax(score)], 100 * np.max(score))
    st.success(string)
