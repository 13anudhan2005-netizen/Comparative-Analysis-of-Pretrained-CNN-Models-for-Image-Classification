import streamlit as st
import numpy as np
import time
from PIL import Image
import pandas as pd

from tensorflow.keras.preprocessing import image

# ===============================
# Import Models
# ===============================
from tensorflow.keras.applications import (
    ResNet50, ResNet50V2,
    VGG16, VGG19,
    Xception, InceptionV3,
    MobileNetV2, DenseNet121,
    NASNetMobile, NASNetLarge,
    EfficientNetV2B0
)

from tensorflow.keras.applications.resnet50 import preprocess_input as res50_pre, decode_predictions as res50_dec
from tensorflow.keras.applications.resnet_v2 import preprocess_input as res50v2_pre, decode_predictions as res50v2_dec
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_pre, decode_predictions as vgg16_dec
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_pre, decode_predictions as vgg19_dec
from tensorflow.keras.applications.xception import preprocess_input as xception_pre, decode_predictions as xception_dec
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_pre, decode_predictions as inception_dec
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_pre, decode_predictions as mobile_dec
from tensorflow.keras.applications.densenet import preprocess_input as dense_pre, decode_predictions as dense_dec
from tensorflow.keras.applications.nasnet import preprocess_input as nas_pre, decode_predictions as nas_dec
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as eff_pre, decode_predictions as eff_dec

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="CNN Model Comparison", layout="wide")
st.title("🧠 Image Classification & CNN Model Comparison")

st.markdown("""
This application compares **multiple ImageNet-pretrained CNN models**
based on **prediction confidence, inference time, model size, and parameters**.
""")

uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "png", "jpeg"])

model_name = st.selectbox(
    "🔍 Select CNN Model",
    [
        "ResNet50", "ResNet50V2",
        "VGG16", "VGG19",
        "Xception", "InceptionV3",
        "MobileNetV2", "DenseNet121",
        "NASNetMobile", "NASNetLarge",
        "EfficientNetV2B0"
    ]
)

top_k = st.slider("🎯 Select Top-K Predictions", 1, 10, 5)

# ===============================
# Load Model (Cached)
# ===============================
@st.cache_resource
def load_model(name):
    if name == "ResNet50":
        return ResNet50(weights="imagenet"), res50_pre, res50_dec, (224,224)
    if name == "ResNet50V2":
        return ResNet50V2(weights="imagenet"), res50v2_pre, res50v2_dec, (224,224)
    if name == "VGG16":
        return VGG16(weights="imagenet"), vgg16_pre, vgg16_dec, (224,224)
    if name == "VGG19":
        return VGG19(weights="imagenet"), vgg19_pre, vgg19_dec, (224,224)
    if name == "Xception":
        return Xception(weights="imagenet"), xception_pre, xception_dec, (299,299)
    if name == "InceptionV3":
        return InceptionV3(weights="imagenet"), inception_pre, inception_dec, (299,299)
    if name == "MobileNetV2":
        return MobileNetV2(weights="imagenet"), mobile_pre, mobile_dec, (224,224)
    if name == "DenseNet121":
        return DenseNet121(weights="imagenet"), dense_pre, dense_dec, (224,224)
    if name == "NASNetMobile":
        return NASNetMobile(weights="imagenet"), nas_pre, nas_dec, (224,224)
    if name == "NASNetLarge":
        return NASNetLarge(weights="imagenet"), nas_pre, nas_dec, (331,331)
    if name == "EfficientNetV2B0":
        return EfficientNetV2B0(weights="imagenet"), eff_pre, eff_dec, (224,224)

# ===============================
# Prediction
# ===============================
if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    st.image(img_pil, caption="Uploaded Image", width=300)

    model, preprocess_fn, decode_fn, size = load_model(model_name)

    img = img_pil.resize(size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_fn(img_array)

    start = time.time()
    preds = model.predict(img_array)
    end = time.time()

    decoded = decode_fn(preds, top=top_k)[0]

    # ===============================
    # Predictions
    # ===============================
    st.subheader("🔍 Top Predictions")

    labels = []
    scores = []

    for i, (_, label, score) in enumerate(decoded):
        labels.append(label)
        scores.append(score)
        st.write(f"**{i+1}. {label}** — {score:.2f}")

    # ===============================
    # Confidence Bar Chart
    # ===============================
    st.subheader("📊 Prediction Confidence")
    chart_data = pd.DataFrame({
        "Label": labels,
        "Confidence": scores
    }).set_index("Label")

    st.bar_chart(chart_data)

    # ===============================
    # Model Comparison Table
    # ===============================
    st.subheader("⚙️ Model Statistics")

    inference_time = (end - start) * 1000
    params = model.count_params()
    size_mb = params * 4 / (1024 ** 2)
    depth = len(model.layers)

    stats_df = pd.DataFrame({
        "Metric": ["Inference Time (ms)", "Parameters", "Model Size (MB)", "Depth"],
        "Value": [
            f"{inference_time:.2f}",
            f"{params:,}",
            f"{size_mb:.2f}",
            depth
        ]
    })

    st.table(stats_df)

    # ===============================
    # Download Results
    # ===============================
    result_df = pd.DataFrame({
        "Label": labels,
        "Confidence": scores
    })

    st.download_button(
        "⬇️ Download Predictions (CSV)",
        data=result_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown("👨‍💻 **Built with TensorFlow, Keras & Streamlit**")
