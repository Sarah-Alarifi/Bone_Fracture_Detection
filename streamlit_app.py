import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
import cv2
import torch
import tensorflow as tf  # Assuming TensorFlow/Keras for the CNN

# Function to load a model
def load_model(model_name: str):
    """
    Load a pre-trained model.

    Args:
        model_name (str): Name of the model file to load.

    Returns:
        The loaded model.
    """
    if model_name.endswith(".pkl"):
        return joblib.load(model_name)
    elif model_name.endswith(".h5"):
        return tf.keras.models.load_model(model_name)
    elif model_name.endswith(".pt"):
        return torch.load(model_name, map_location=torch.device('cpu'))  # Load for CPU by default
    else:
        raise ValueError("Unsupported model file format.")

# Function to preprocess image for CNN
def preprocess_image_for_cnn(img, img_size=(224, 224)):
    """
    Preprocess the image for CNN input.

    Args:
        img (PIL.Image): The input image.
        img_size (tuple): Target size for resizing.

    Returns:
        np.ndarray: Preprocessed image.
    """
    image = img.resize(img_size).convert("RGB")
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Function to classify image using CNN
def classify_with_cnn(img, model):
    """
    Classify the image using the CNN model.

    Args:
        img (PIL.Image): The input image.
        model: The loaded CNN model.

    Returns:
        dict: Class labels and probabilities.
    """
    try:
        preprocessed_img = preprocess_image_for_cnn(img)
        predictions = model.predict(preprocessed_img)
        class_idx = np.argmax(predictions)
        confidence = predictions[0][class_idx]
        return {"class_idx": class_idx, "confidence": confidence}
    except Exception as e:
        st.error(f"An error occurred during CNN classification: {e}")
        return None

# Streamlit app
st.title("Bone Structure Analysis and Object Detection")
st.write("Upload an X-ray or bone scan image for analysis.")

# Upload image
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Model selection
model_type = st.selectbox("Choose a classification model:", ["KNN", "ANN", "SVM", "YOLO Object Detection", "CNN"])

# Load models
if model_type in ["KNN", "ANN", "SVM", "CNN"]:
    try:
        model_files = {
            "KNN": "knn_classifier.pkl",
            "ANN": "ann_classifier.pkl",
            "SVM": "svm_classifier.pkl",
            "CNN": "small_cnn_with_dropout.pkl"
        }
        selected_model_file = model_files[model_type]
        model = load_model(selected_model_file)
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()

# YOLO object detection
if model_type == "YOLO Object Detection":
    try:
        yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='kidney_yolo.pt')  # Replace with your YOLO model
        st.success("YOLO model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    pred_button = st.button("Analyze Image")

    if pred_button:
        if model_type in ["KNN", "ANN", "SVM"]:
            # Perform classification
            predictions_df, top_prediction = classify_image(image_file, model, model_type)
            if not predictions_df.empty:
                st.success(f'Predicted Structure: **{top_prediction}** '
                           f'Confidence: {predictions_df.iloc[0]["Probability"]:.2%}')
                st.write("Detailed Predictions:")
                st.table(predictions_df)
            else:
                st.error("Failed to classify the image.")

        elif model_type == "CNN":
            # Perform CNN classification
            image = Image.open(image_file).convert("RGB")
            cnn_result = classify_with_cnn(image, model)
            if cnn_result:
                st.success(f"Predicted Class: {cnn_result['class_idx']}, Confidence: {cnn_result['confidence']:.2%}")
            else:
                st.error("Failed to classify the image with CNN.")

        elif model_type == "YOLO Object Detection":
            # Perform YOLO detection
            image = Image.open(image_file).convert("RGB")
            image_np = np.array(image)
            results = yolo_model(image_np)
            detection_data = results.pandas().xyxy[0]
            if detection_data.empty:
                st.warning("No objects detected in the image.")
            else:
                st.image(results.render()[0], caption="YOLO Detection Results", use_column_width=True)
                st.write("Detection Results:")
                detection_data = detection_data[["xmin", "ymin", "xmax", "ymax", "confidence"]]
                st.table(detection_data)
