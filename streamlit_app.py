import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
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
        # Preprocess the image
        image = Image.open(img).convert("RGB")
        image = image.resize((224, 224))  # Resize to match expected input
        image_array = np.array(image).flatten()  # Flatten the image for KNN

        # Debugging: Print the shape of the processed image
        st.write(f"Processed image shape: {image_array.shape}")
        st.write(f"Scaler expected shape: {scaler.mean_.shape}")

        # Scale the image features
        image_array = scaler.transform([image_array])  # Ensure correct input shape

        # Predict using the KNN model
        prediction = model.predict(image_array)
        probabilities = model.predict_proba(image_array)[0]  # Get class probabilities

        # Create a DataFrame to store predictions and probabilities
        class_labels = model.classes_  # Get class labels from the model
        prediction_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": probabilities
        })
        return prediction_df.sort_values("Probability", ascending=False)

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return pd.DataFrame()

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
        predictions_df = classify_image_knn(image_file, knn_model, scaler)

        if not predictions_df.empty:
            # Display top prediction
            top_prediction_row = predictions_df.iloc[0]
            st.success(f'Predicted Structure: **{top_prediction_row["Class"]}** '
                       f'Confidence: {top_prediction_row["Probability"]:.2%}')

            # Display all predictions
            st.write("Detailed Predictions:")
            st.table(predictions_df)
        else:
            st.error("Failed to classify the image.")
