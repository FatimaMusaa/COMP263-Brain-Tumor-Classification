from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image

from config import CLASS_NAMES, DEFAULT_MODEL_NAME, IMAGE_SIZE, MODELS_DIR, OUTPUTS_DIR
from model_factory import create_model, list_supported_models


st.set_page_config(
    page_title="Brain Tumor MRI Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(0, 153, 255, 0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 255, 170, 0.12), transparent 24%),
            linear-gradient(180deg, #07111f 0%, #0b1728 55%, #050b14 100%);
        color: #e8eef8;
    }

    [data-testid="stAppViewContainer"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1626 0%, #09111d 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    [data-testid="stHeader"] {
        background: rgba(5, 11, 20, 0.75);
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: #e8eef8;
    }

    .stMarkdown a {
        color: #7dd3fc;
    }

    [data-testid="stMetric"] {
        background: rgba(13, 24, 40, 0.82);
        border: 1px solid rgba(125, 211, 252, 0.18);
        border-radius: 16px;
        padding: 0.85rem;
    }

    [data-testid="stFileUploader"],
    [data-testid="stExpander"],
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(10, 19, 33, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        min-height: 2.8rem;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #0369a1 100%);
        color: #ffffff !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border-color: rgba(148, 163, 184, 0.18) !important;
    }

    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] svg,
    .stSelectbox div[data-baseweb="select"] input {
        color: #000000 !important;
        fill: #000000 !important;
        caret-color: #000000 !important;
    }

    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] span {
        color: #000000 !important;
        background: #ffffff !important;
    }

    div[data-baseweb="popover"] * {
        color: #000000 !important;
    }

    [data-testid="stFileUploader"] section {
        background: #ffffff !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] label {
        color: #000000 !important;
    }

    [data-testid="stFileUploader"] button {
        background: #0ea5e9 !important;
        color: #ffffff !important;
        border: none !important;
    }

    [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    [data-testid="stFileUploader"] button svg {
        fill: #ffffff !important;
    }

    .stAlert {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_LABELS = {
    "baseline_cnn": "Baseline CNN",
    "efficientnet_b0_transfer": "EfficientNetB0 Transfer",
    "densenet121_transfer": "DenseNet121 Transfer",
}

TUMOR_INFO = {
    "glioma": {
        "title": "Glioma",
        "definition": (
            "Glioma is a type of brain tumor that starts in glial cells, which "
            "support and protect nerve cells in the brain."
        ),
        "link_label": "Mayo Clinic: Glioma overview",
        "link_url": "https://www.mayoclinic.org/diseases-conditions/glioma/symptoms-causes/syc-20350251",
    },
    "meningioma": {
        "title": "Meningioma",
        "definition": (
            "Meningioma is a tumor that begins in the meninges, the membranes "
            "that surround the brain and spinal cord. Many are slow-growing and benign."
        ),
        "link_label": "Mayo Clinic: Meningioma overview",
        "link_url": "https://www.mayoclinic.org/diseases-conditions/meningioma/symptoms-causes/syc-20355643",
    },
    "notumor": {
        "title": "No Tumor",
        "definition": (
            "This class means the model did not find image features matching the tumor "
            "categories it was trained on. It does not replace a radiologist's review."
        ),
        "link_label": "Mayo Clinic: Brain tumor overview",
        "link_url": "https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/syc-20350084",
    },
    "pituitary": {
        "title": "Pituitary Tumor",
        "definition": (
            "A pituitary tumor is an unusual growth in the pituitary gland at the base "
            "of the brain. Many are benign, but they can affect hormone levels or press on nearby structures."
        ),
        "link_label": "Mayo Clinic: Pituitary tumors overview",
        "link_url": "https://www.mayoclinic.org/health/pituitary-tumors/DS00533",
    },
}


def format_label(label: str) -> str:
    return "No Tumor" if label == "notumor" else label.capitalize()


def available_model_files() -> list[str]:
    return [
        model_name
        for model_name in list_supported_models()
        if (MODELS_DIR / f"{model_name}.keras").exists()
    ]


def load_model_comparison() -> list[dict]:
    comparison_path = OUTPUTS_DIR / "model_comparison.json"
    if comparison_path.exists():
        with open(comparison_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return []


def load_model_metrics(model_name: str) -> dict | None:
    metrics_path = OUTPUTS_DIR / f"{model_name}_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return None


@st.cache_resource
def load_trained_model(model_name: str):
    bundle = create_model(model_name)
    bundle.model.load_weights(MODELS_DIR / f"{model_name}.keras")
    return bundle


def preprocess_image(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.array(rgb_image).astype("float32")
    return np.expand_dims(image_array, axis=0)


def classify_prediction(predictions: np.ndarray) -> tuple[str, str]:
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    predicted_prob = float(predictions[predicted_index])

    if predicted_prob < 0.55:
        return predicted_class, "Low confidence"

    if predicted_class == "notumor":
        return predicted_class, "No tumor detected"

    return predicted_class, "Tumor detected"


def render_tumor_info(label_key: str) -> None:
    info = TUMOR_INFO[label_key]
    st.markdown(f"### About {info['title']}")
    st.write(info["definition"])
    st.markdown(f"[{info['link_label']}]({info['link_url']})")


def get_last_feature_layer(model: tf.keras.Model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            output_shape = getattr(layer, "output_shape", None)
            if output_shape is not None and len(output_shape) == 4:
                return layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None


def make_gradcam_overlay(
    bundle,
    processed_image: np.ndarray,
    original_image: Image.Image,
    class_index: int,
) -> Image.Image | None:
    feature_layer = get_last_feature_layer(bundle.model)
    if feature_layer is None:
        return None

    try:
        grad_model = tf.keras.models.Model(
            inputs=bundle.model.inputs,
            outputs=[feature_layer.output, bundle.model.output],
        )

        input_tensor = tf.convert_to_tensor(processed_image)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor)
            target_score = predictions[:, class_index]

        grads = tape.gradient(target_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)

        max_val = tf.reduce_max(heatmap)
        if float(max_val) == 0.0:
            return None

        heatmap = heatmap / max_val
        heatmap = heatmap.numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap_image = Image.fromarray(heatmap).resize(original_image.size)

        heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0
        colored = colormaps["jet"](heatmap_array)[..., :3]
        original = np.asarray(original_image.convert("RGB"), dtype=np.float32) / 255.0
        overlay = np.clip((0.55 * original) + (0.45 * colored), 0, 1)
        return Image.fromarray(np.uint8(overlay * 255))
    except Exception:
        return None


available_models = available_model_files()
default_index = (
    available_models.index(DEFAULT_MODEL_NAME)
    if DEFAULT_MODEL_NAME in available_models
    else 0
)

with st.sidebar:
    st.title("MRI Tumor Detector")
    st.caption("Deep learning demo for COMP 263")

    st.markdown("## 🧠 MRI Detector")

    if not available_models:
        st.error("No trained model files were found in the models folder.")
        st.stop()

    selected_model_name = st.selectbox(
        "Choose model",
        options=available_models,
        index=default_index,
        format_func=lambda name: MODEL_LABELS.get(name, name),
    )

    st.markdown("---")

    # Clean educational note
    st.caption(
        "Group 6 Project • COMP 263\n\n"
        "This tool is for educational purposes only and "
        "is not intended for medical diagnosis."
    )

bundle = load_trained_model(selected_model_name)

st.title("Brain Tumor MRI Detection and Classification")
st.write(
    "Upload an MRI image to predict whether a tumor is present and, if so, "
    "which tumor type is most likely."
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Upload MRI")
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG brain MRI image",
        type=["jpg", "jpeg", "png"],
    )

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI", use_container_width=500)
    else:
        st.info("Upload an image to begin.")

with right_col:
    st.subheader("Prediction")
    analyze_clicked = st.button("Analyze MRI", use_container_width=True)

    if analyze_clicked and image is None:
        st.warning("Please upload an MRI image before running the analysis.")
    elif analyze_clicked and image is not None:
        with st.spinner("Running inference..."):
            processed = preprocess_image(image)
            predictions = bundle.model.predict(processed, verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            predicted_class, status_text = classify_prediction(predictions)
            overlay = make_gradcam_overlay(bundle, processed, image, predicted_index)

        confidence = float(predictions[predicted_index])
        top_indices = np.argsort(predictions)[::-1]

 # 🎯 COLOR RESULT DISPLAY

        if predicted_class == "notumor" and confidence >= 0.55:
          st.success("🟢 No Tumor Detected")

        elif confidence < 0.55:
          st.warning("🟡 Low Confidence Prediction")

        else:
         st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #7f1d1d, #991b1b);
            padding: 14px;
            border-radius: 12px;
            color: white;
            font-weight: bold;
            font-size: 20px;
            text-align: center;
        ">
        🔴 {format_label(predicted_class)} Tumor Detected
        </div>
        """,
        unsafe_allow_html=True
    )
        if predicted_class != "notumor":
         st.markdown(f"### 🔴 {format_label(predicted_class)}")
        else:
         st.markdown("### 🟢 No Tumor")
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.caption(f"Model used: {MODEL_LABELS.get(selected_model_name, selected_model_name)}")

        st.markdown("### Class probabilities")
        for index in top_indices:
            label = format_label(CLASS_NAMES[int(index)])
            prob = float(predictions[int(index)])
            st.markdown(
                f"<div style='font-size:16px; font-weight:600;'>"
                f"{label}: {prob * 100:.2f}%"
                f"</div>",
                unsafe_allow_html=True
                )
            st.progress(prob)
            

        render_tumor_info(predicted_class)

        if overlay is not None:
            st.markdown("### Visual explanation")
            st.image(
                overlay,
                caption="Grad-CAM style heatmap overlay showing influential regions",
                use_container_width=500,
            )
        else:
            st.caption("Visual explanation is not available for this saved model build.")
    else:
        st.info("Prediction results will appear here.")


st.markdown("---")
st.write(
    "Recommended submission model: EfficientNetB0 transfer learning. "
    "It offers the best balance of accuracy, generalization, and runtime for this dataset."
)

with st.expander("Tumor Definitions and Official References"):
    for class_name in CLASS_NAMES:
        info = TUMOR_INFO[class_name]
        st.markdown(f"**{info['title']}**")
        st.write(info["definition"])
        st.markdown(f"[{info['link_label']}]({info['link_url']})")

st.markdown("---")


with st.expander("📊 Model Evaluation Details (Click to View)"):

    metrics = load_model_metrics(selected_model_name)

    if metrics is None:
        st.info("No saved evaluation file was found for the selected model yet.")
    else:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with metric_col2:
            st.metric("Macro Precision", f"{metrics['macro_precision']:.3f}")
        with metric_col3:
            st.metric("Macro Recall", f"{metrics['macro_recall']:.3f}")
        with metric_col4:
            st.metric("Macro F1", f"{metrics['macro_f1']:.3f}")

        # ✅ CONTROL CONFUSION MATRIX DISPLAY
        show_cm = st.checkbox("Show Confusion Matrix")

        if show_cm:
            st.markdown("### Confusion Matrix")
            st.caption("Rows are actual classes and columns are predicted classes.")

            matrix_image_path = OUTPUTS_DIR / f"{selected_model_name}_confusion_matrix.png"

            if matrix_image_path.exists():
                st.image(
                    str(matrix_image_path),
                    caption="Confusion matrix visualization",
                    width=500
                )

      # ✅ Tabs ALSO INSIDE expander
details_tab, report_tab, matrix_tab = st.tabs(
    ["Model Info", "Classification Report", "Confusion Matrix Table"]
)

# ------------------ DETAILS TAB ------------------
with details_tab:
    st.write(f"**Model name:** {MODEL_LABELS.get(selected_model_name, selected_model_name)}")
    st.write(f"**Classes:** {', '.join(format_label(name) for name in CLASS_NAMES)}")
    st.write(f"**Input size:** {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")

# ------------------ REPORT TAB ------------------
with report_tab:
    st.markdown("### Classification Report")

    headers = st.columns([1.5, 1, 1, 1, 1])
    headers[0].write("**Class**")
    headers[1].markdown("<div style='text-align:center'><b>Precision</b></div>", unsafe_allow_html=True)
    headers[2].markdown("<div style='text-align:center'><b>Recall</b></div>", unsafe_allow_html=True)
    headers[3].markdown("<div style='text-align:center'><b>F1-score</b></div>", unsafe_allow_html=True)
    headers[4].markdown("<div style='text-align:center'><b>Support</b></div>", unsafe_allow_html=True)

    for class_name in CLASS_NAMES:
        class_report = metrics["classification_report"].get(class_name, {})

        cols = st.columns([1.5, 1, 1, 1, 1])

        cols[0].write(f"**{format_label(class_name)}**")

        cols[1].markdown(
            f"<div style='text-align:center'><b>{round(class_report.get('precision', 0.0), 3)}</b></div>",
            unsafe_allow_html=True
        )
        cols[2].markdown(
            f"<div style='text-align:center'><b>{round(class_report.get('recall', 0.0), 3)}</b></div>",
            unsafe_allow_html=True
        )
        cols[3].markdown(
            f"<div style='text-align:center'><b>{round(class_report.get('f1-score', 0.0), 3)}</b></div>",
            unsafe_allow_html=True
        )
        cols[4].markdown(
            f"<div style='text-align:center'><b>{int(class_report.get('support', 0))}</b></div>",
            unsafe_allow_html=True
        )

# ------------------ MATRIX TAB ------------------
with matrix_tab:
    st.markdown("### Confusion Matrix (Table View)")
    st.caption("Rows = Actual | Columns = Predicted")

    cm = np.array(metrics["confusion_matrix"])

    header = st.columns([1.5, 1, 1, 1, 1])
    header[0].write("**Actual \\ Predicted**")

    for i, name in enumerate(CLASS_NAMES):
        header[i+1].markdown(
            f"<div style='text-align:center'><b>{format_label(name)}</b></div>",
            unsafe_allow_html=True
        )

    for row_index, actual_name in enumerate(CLASS_NAMES):
        cols = st.columns([1.5, 1, 1, 1, 1])

        cols[0].write(f"**{format_label(actual_name)}**")

        for col_index in range(len(CLASS_NAMES)):
            value = int(cm[row_index, col_index])

            cols[col_index+1].markdown(
                f"<div style='text-align:center'><b>{value}</b></div>",
                unsafe_allow_html=True
            )