import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Set up the page
st.set_page_config(page_title="NeuroScan Lite", page_icon="🧠")
st.title("🧠 NeuroScan Lite")
st.write("Secure, Offline-Ready Brain Tumor Detection for Edge Devices.")

# Load the TinyML Model
@st.cache_resource
def load_model():
    # Make sure 'tumor_detector_quantized.tflite' is in the same folder!
    interpreter = tf.lite.Interpreter(model_path="tumor_detector_quantized.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("✅ System Ready: Quantized MobileNetV2 Loaded")
except:
    st.error("Model missing. Please run the research notebook first.")

# Upload & Analyze
uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Scan Uploaded', use_column_width=True)
    
    # Preprocess image to fit the AI (224x224)
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    if st.button("Run Diagnostic"):
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        confidence = output_data[0][0]
        
        st.write("---")
        if confidence > 0.5:
            st.error(f"🚨 RESULT: Tumor Detected ({confidence*100:.1f}% Confidence)")
        else:
            st.success(f"✅ RESULT: Healthy ({ (1-confidence)*100:.1f}% Confidence)")
            
        st.info("Note: This model is optimized for speed (TinyML). Always verify with a doctor.")