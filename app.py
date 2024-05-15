from PIL import Image
import streamlit as st
import cv2
#import urllib.request
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('best_model.keras')
  return model

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
