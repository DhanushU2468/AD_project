import streamlit as st
import torch, numpy as np, cv2
from PIL import Image
from models.hybrid_model import HybridModel
from explainability.gradcam import GradCAM
from skimage.transform import resize
import torch.nn.functional as F

st.set_page_config(page_title="🧠 Alzheimer’s MRI Classifier", layout="centered")
st.title("🧠 Alzheimer’s Disease Classification with Grad-CAM")
st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress pyplot global use warning

labels = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridModel(num_classes=4).to(device)
    model.load_state_dict(torch.load("models/model_epoch_2.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

uploaded = st.file_uploader("📤 Upload an MRI image (.jpg)", type=["jpg", "png", "jpeg"])

if uploaded:
    # ---- Load and preprocess ----
    img = Image.open(uploaded).convert("L")
    img_arr = np.array(img)
    img_resized = resize(img_arr, (128, 128))
    x = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # ---- Find last convolutional layer ----
    target_layer = None
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            target_layer = layer
    gradcam = GradCAM(model, target_layer)

    # ---- Forward pass ----
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        idx = np.argmax(probs)
        pred_label = labels[idx]

    # ---- Grad-CAM heatmap ----
    cam, _ = gradcam.generate(x)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig = cv2.cvtColor(np.uint8(img_resized * 255), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    # ---- Display Results ----
    st.image([img, overlay], caption=["Original MRI", f"Prediction: {pred_label}"], width=300)
    st.success(f"🧠 Predicted: **{pred_label}**")

    # ---- Display Probability Bar Chart ----
    st.subheader("📊 Prediction Confidence")
    st.bar_chart({labels[i]: probs[i] for i in range(len(labels))})
