## 3️⃣ Develop Streamlit Dashboard

import streamlit as st
from PIL import Image
import numpy as np

# Load trained model
model = keras.models.load_model('waste_classifier.h5')

def preprocess_image(image):
    img = image.resize((img_height, img_width))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

st.title("♻️ AI-Powered Waste Classification")
st.write("Upload an image to classify waste type")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.write(f"### 🏷️ Predicted Waste Category: {predicted_class}")
    st.write(f"### ✅ Confidence: {confidence:.2f}%")
    
    # Show prediction probability as a bar chart
    st.bar_chart(prediction.flatten())