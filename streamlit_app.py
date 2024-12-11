import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
import cv2  # For SIFT feature extraction
import torch

# Function to load a model
def load_model(model_name: str):
    """
    Load a pre-trained model.

    Args:
        model_name (str): Name of the model file to load.

    Returns:
        sklearn.base.BaseEstimator: The loaded model.
    """
    return joblib.load(model_name)

# Function to extract SIFT features
def extract_features(img) -> np.ndarray:
    """
    Extract features from the image using SIFT.

    Args:
        img (PIL.Image): The input image.

    Returns:
        np.ndarray: Feature vector of fixed size (128).
    """
    image_cv = np.array(img)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)

    if descriptors is not None:
        return descriptors.flatten()[:128]  # Truncate/pad to fixed size
    else:
        return np.zeros(128)  # Zero vector if no features are found

# Function to preprocess and classify an image
def classify_image(img: bytes, model, model_type: str) -> pd.DataFrame:
    """
    Classify the given image using the selected model and return predictions.

    Args:
        img (bytes): The image file to classify.
        model: The pre-trained model.
        model_type (str): The type of model (KNN, ANN, or SVM).

    Returns:
        pd.DataFrame: A DataFrame containing predictions and their probabilities.
    """
    try:
        image = Image.open(img).convert("RGB")
        features = extract_features(image)

        # Predict based on the model type
        if model_type == "KNN" or model_type == "SVM":
            prediction = model.predict([features])
            probabilities = model.predict_proba([features])[0]  # Class probabilities
        elif model_type == "ANN":
            probabilities = model.predict_proba([features])[0]  # For ANN
            prediction = [np.argmax(probabilities)]  # Get class with highest probability

        # Map numeric predictions to descriptive labels
        LABEL_MAPPING = {
            0: "Not Fractured",
            1: "Fractured"
        }
        class_labels = [LABEL_MAPPING[cls] for cls in model.classes_]

        # Create a DataFrame to store predictions and probabilities
        prediction_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": probabilities
        })
        return prediction_df.sort_values("Probability", ascending=False), LABEL_MAPPING[prediction[0]]

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return pd.DataFrame(), None

# Streamlit app
st.title("Bone Structure Analysis and Object Detection")
st.write("Upload an X-ray or bone scan image for analysis.")

# Upload image
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Model selection
model_type = st.selectbox("Choose a classification model:", ["KNN", "ANN", "SVM", "YOLO Object Detection","cnn_with_dropout"])

# Load classification models
if model_type in ["KNN", "ANN", "SVM"]:
    try:
        model_files = {
            "KNN": "knn_classifier.pkl",
            "ANN": "ann_classifier.pkl",
            "SVM": "svm_classifier.pkl",
            "cnn_with_dropout": "small_cnn_with_dropout.pkl"
        }
        selected_model_file = model_files[model_type]
        model = load_model(selected_model_file)
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()

# YOLO object detection
if model_type == "YOLO Object Detection":
    try:
        # Load YOLO model
        yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='kidney_yolo.pt')  # Replace 'best.pt' with your YOLO model path
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
                # Display top prediction
                st.success(f'Predicted Structure: **{top_prediction}** '
                           f'Confidence: {predictions_df.iloc[0]["Probability"]:.2%}')
                # Display all predictions
                st.write("Detailed Predictions:")
                st.table(predictions_df)
            else:
                st.error("Failed to classify the image.")

        elif model_type == "YOLO Object Detection":
            # Load and preprocess image
            image = Image.open(image_file).convert("RGB")
            image_np = np.array(image)

            # Perform object detection
            results = yolo_model(image_np)

            # Filter results for the single class
            detection_data = results.pandas().xyxy[0]
            if detection_data.empty:
                st.warning("No objects detected in the image.")
            else:
                # Display results with bounding boxes
                st.image(results.render()[0], caption="YOLO Detection Results", use_column_width=True)

                # Display detection confidence
                st.write("Detection Results:")
                detection_data = detection_data[["xmin", "ymin", "xmax", "ymax", "confidence"]]
                st.table(detection_data)
