import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
import cv2  # For SIFT feature extraction
from sklearn.preprocessing import StandardScaler

# Function to load the KNN classifier model
def load_knn_model(model_name: str = "knn_classifier.pkl"):
    """
    Load the pre-trained KNN model.

    Args:
        model_name (str): Name of the model file to load.

    Returns:
        sklearn.base.BaseEstimator: The loaded KNN model.
    """
    knn_model = joblib.load(model_name)
    return knn_model

# Function to extract SIFT features
def extract_features(img) -> np.ndarray:
    """
    Extract features from the image using SIFT.

    Args:
        img (PIL.Image): The input image.

    Returns:
        np.ndarray: Feature vector of fixed size (128).
    """
    # Convert PIL image to OpenCV format
    image_cv = np.array(img)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    # Use SIFT to extract features
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)

    # Flatten descriptors to a fixed size
    if descriptors is not None:
        return descriptors.flatten()[:128]  # Ensure consistent size (truncate/pad)
    else:
        return np.zeros(128)  # Return a zero vector if no features are found

# Mapping numeric labels to descriptive names
LABEL_MAPPING = {
    0: "Not Fractured",
    1: "Fractured"
}

# Function to preprocess and classify an image
def classify_image_knn(img: bytes, model, scaler) -> pd.DataFrame:
    """
    Classify the given image using the KNN model and return predictions.

    Args:
        img (bytes): The image file to classify.
        model: The pre-trained KNN model.
        scaler: The scaler used to preprocess features.

    Returns:
        pd.DataFrame: A DataFrame containing predictions and their probabilities.
    """
    try:
        # Preprocess the image and extract features
        image = Image.open(img).convert("RGB")
        features = extract_features(image)

        # Debugging: Print feature shape
        st.write(f"Extracted feature vector shape: {features.shape}")
        st.write(f"Scaler expected shape: {scaler.mean_.shape}")

        # Scale the features
        features_scaled = scaler.transform([features])  # Ensure correct input shape

        # Predict using the KNN model
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]  # Get class probabilities

        # Map numeric predictions to descriptive labels
        class_labels = [LABEL_MAPPING[cls] for cls in model.classes_]

        # Create a DataFrame to store predictions and probabilities
        prediction_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": probabilities
        })
        return prediction_df.sort_values("Probability", ascending=False), LABEL_MAPPING[prediction]

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return pd.DataFrame(), None

# Streamlit app
st.title("Bone Structure Analysis")
st.write("Upload an X-ray or bone scan image to analyze the structure.")

# Upload image
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Load the pre-trained KNN model and scaler
try:
    knn_model = load_knn_model("knn_classifier.pkl")
    scaler = joblib.load("scaler.pkl")  # Ensure the scaler used for training is saved and loaded
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    pred_button = st.button("Analyze Bone Structure")
    
    if pred_button:
        # Perform image classification
        predictions_df, top_prediction = classify_image_knn(image_file, knn_model, scaler)

        if not predictions_df.empty:
            # Display top prediction
            st.success(f'Predicted Structure: **{top_prediction}** '
                       f'Confidence: {predictions_df.iloc[0]["Probability"]:.2%}')

            # Display all predictions
            st.write("Detailed Predictions:")
            st.table(predictions_df)
        else:
            st.error("Failed to classify the image.")
