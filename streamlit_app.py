import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf  # إذا كنت تستخدم TensorFlow

# تحميل النموذج
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/your_model_file.h5')  # مسار النموذج
    return model

# تحليل الصورة
def predict_image(model, image):
    # تغيير حجم الصورة بما يتناسب مع النموذج
    image_resized = cv2.resize(image, (224, 224))  # تأكد من أن هذا الحجم مناسب لنموذجك
    image_array = np.expand_dims(image_resized, axis=0)  # إضافة بعد جديد للدفعة
    prediction = model.predict(image_array)
    return prediction.tolist()

# واجهة Streamlit
st.title("تحليل الصور باستخدام نموذج AI")

st.write("ارفع صورة لتحليلها باستخدام النموذج الخاص بك.")

# رفع صورة
uploaded_file = st.file_uploader("اختر صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة المرفوعة", use_column_width=True)

    # تحويل الصورة إلى صيغة numpy
    image_array = np.array(image)

    # تحميل النموذج وتحليل الصورة
    model = load_model()
    result = predict_image(model, image_array)

    # عرض النتائج
    st.write("### النتيجة")
    st.write(result)
