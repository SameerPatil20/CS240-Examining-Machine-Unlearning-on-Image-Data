import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Results Viewer", page_icon="🔍")

st.title("Results Viewer")
st.write("Select a model from the dropdown to view its confusion matrix and classification report.")

image_files = {
    "Baseline": {
        "confusion": "static/cm_10res18_baseline.png",
        "classification": "static/cr_10res18_baseline.png",
        "Accuracy": "76%"
    },
    "Retrained": {
        "confusion": "static/cm_10res18_retrain.png",
        "classification": "static/cr_10res18_retrain.png",
        "Accuracy": "65%"
    },
    "SSD": {
        "confusion": "static/cm_10res18_ssd.png",
        "classification": "static/cr_10res18_ssd.png",
        "Accuracy": "64%"
    },
    "DeepClean": {
        "confusion": "static/cm_10res18_deepclean.png",
        "classification": "static/cr_10res18_deepclean.png",
        "Accuracy": "61%"
    },
    "Incompetent Teacher": {
        "confusion": "static/cm_10res18_teacher.png",
        "classification": "static/cr_10res18_teacher.png",
        "Accuracy": "67%"
    },
    "SalUn": {
        "confusion": "static/cm_10res18_salun.png",
        "classification": "static/cr_10res18_salun.png",
        "Accuracy": "59%"
    }
}

model_choice = st.selectbox("Choose model", list(image_files.keys()))

accuracy_value = image_files[model_choice]["Accuracy"]
confusion_path = image_files[model_choice]["confusion"]
classification_path = image_files[model_choice]["classification"]

st.write(f"### Model Accuracy: {accuracy_value}")
col1, col2 = st.columns(2)
if Path(confusion_path).exists():
    col1.image(confusion_path, caption="Confusion Matrix", use_container_width=True)
else:
    col1.warning(f"Confusion matrix image not found: {confusion_path}")

if Path(classification_path).exists():
    col2.image(classification_path, caption="Classification Report", use_container_width=True)
else:
    col2.warning(f"Classification report image not found: {classification_path}")
