import tensorflow as tf
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np

# Function to load the bone structure model
def load_model(model_name: str = "yolo_kidney.h5") -> tf.keras.Model:
    """
    Load the pre-trained bone structure model.
    
    Args:
        model_name (str): Name of the model file to load.
    
    Returns:
        tf.keras.Model: The loaded TensorFlow model.
    """
    tf_model = tf.keras.models.load_model(f"models_and_data/{model_name}")
    return tf_model

# Function to classify a bone structure image
def classify_image(img: bytes, model: tf.keras.Model) -> pd.DataFrame:
    """
    Classify the given image using the provided model and return predictions.
    
    Args:
        img (bytes): The image file to classify.
        model (tf.keras.Model): The pre-trained model to use for prediction.
    
    Returns:
        pd.DataFrame: A DataFrame containing predictions and their probabilities.
    """
    # Preprocess the image
    image = Image.open(img).convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make predictions using the model
    pred_probs = model.predict(image_array)[0]  # Get predictions
    class_indices = np.argsort(pred_probs)[::-1][:3]  # Top 3 predictions
    class_labels = ["Tas_Var"]  # Replace with actual class names
    probabilities = pred_probs[class_indices]

    # Create a DataFrame to store predictions and probabilities
    prediction_df = pd.DataFrame({
        "Class": [class_labels[i] for i in class_indices],
        "Probability": probabilities
    })
    return prediction_df.sort_values("Probability", ascending=False)

# Streamlit app
st.title("Bone Structure Analysis")
st.write("Upload an X-ray or bone scan image to analyze the structure.")

# Upload image
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    pred_button = st.button("Analyze Bone Structure")
    
    if pred_button:
        # Load the model
        model = load_model()
        
        # Perform image classification
        predictions_df = classify_image(image_file, model)
        
        # Display top prediction
        top_prediction_row = predictions_df.iloc[0]
        st.success(f'Predicted Structure: **{top_prediction_row["Class"]}** '
                   f'Confidence: {top_prediction_row["Probability"]:.2%}')
        
        # Display all predictions
        st.write("Detailed Predictions:")
        st.table(predictions_df)

# Note: Replace "Class A", "Class B", "Class C" with actual bone structure class names.
