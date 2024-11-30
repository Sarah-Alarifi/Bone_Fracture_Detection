import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf  
import joblib

@st.cache_resource
def load_model():
    model = joblib.load('ann.joblib') 
    return model

def predict_image(model, image):
    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)  
    prediction = model.predict(image_array)
    return prediction.tolist()

st.title("Bone Fracture Detection")

st.write("upload the photo")

uploaded_file = st.file_uploader("choose", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="photo", use_column_width=True)

    image_array = np.array(image)

    model = load_model()
    result = predict_image(model, image_array)

    st.write("### result")
    st.write(result)
