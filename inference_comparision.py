import streamlit as st
import torch
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Inference Comparison", page_icon="📊")

st.title("Inference Comparison")
st.write("Select a test CIFAR-10 image and compare predictions of all six models.")
st.write("The classes 0(Airplane) and 1(Automobile) have been unlearnt by the models")

testset = CIFAR10(root=".", train=False, download=False)

class_names = testset.classes

selection_method = st.radio("Select image by:", ("Index", "Gallery"))

if selection_method == "Index":
    index = st.number_input(
        "Enter CIFAR-10 test image index (0-9999):",
        min_value=0, max_value=len(testset)-1, value=0, step=1
    )
else:
    st.write("### Preset Gallery Images")
    preset_indices = [36, 85, 122, 357, 5342, 9876]
    cols1 = st.columns(3)
    for i, idx in enumerate(preset_indices[:3]):
        img, label = testset[idx]
        cols1[i].image(img, caption=f"Index {idx}: {class_names[label]}", use_container_width=True)
    cols2 = st.columns(3)
    for i, idx in enumerate(preset_indices[3:]):
        img, label = testset[idx]
        cols2[i].image(img, caption=f"Index {idx}: {class_names[label]}", use_container_width=True)
    index = st.selectbox("Select an image index from the gallery:", preset_indices)

if 'index' in locals():
    img, true_label = testset[index]
    st.image(
        img,
        caption=f"Selected Image (Index {index}, True Label: {class_names[true_label]})",
        use_container_width=True
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])
    img_tensor = transform(img).unsqueeze(0)

    model_paths = {
        "All Data": "models/10res18_baseline.pth",
        "Retrained": "models/10res18_retrain.pth",
        "SSD": "models/10res18_ssd.pth",
        "DeepClean": "models/10res18_deepclean.pth",
        "Incompetent Teacher": "models/10res18_teacher.pth",
        "SalUn": "models/10res18_salun.pth"
    }

    results = []
    for name, path in model_paths.items():
        if not Path(path).exists():
            st.warning(f"Model file not found: {path}")
            continue
        net = torchvision.models.resnet18(num_classes=10)
        net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        net.eval()

        with torch.no_grad():
            outputs = net(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
            predicted_label = class_names[top_idx.item()]
            prob_percent = top_prob.item() * 100.0

        results.append({
            "Model": name,
            "Predicted Class": predicted_label,
            "Probability": f"{prob_percent:.2f}%"
        })

    if results:
        df = pd.DataFrame(results)
        st.write("### Prediction Results")
        st.table(df)
    else:
        st.write("No models were loaded for inference. Please check model paths.")
