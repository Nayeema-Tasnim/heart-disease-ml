import streamlit as st
from PIL import Image
from torchvision import transforms
import torch

# project imports
from src.tabular.infer import predict_one as predict_tabular
from src.imaging.infer import load_model as load_cnn

st.set_page_config(page_title='Heart Disease (Multimodal)', layout='wide')
st.title('Heart Disease Prediction — Multimodal')

# ---- Caching helpers (faster & cleaner) ----
@st.cache_resource(show_spinner=False)
def get_cnn():
    # Expecting: returns (model, device, img_size)
    return load_cnn()

@st.cache_data(show_spinner=False)
def load_uci():
    import pandas as pd
    return pd.read_csv("data/uci/heart_cleveland.csv")

@st.cache_resource(show_spinner=False)
def load_tabular_model():
    import joblib
    from src.common.paths import TABULAR_MODELS
    return joblib.load(TABULAR_MODELS / "best_model.joblib")

# ---- Tabs (single group) ----
tab1, tab2, tab3 = st.tabs(["Clinical (Tabular)", "Chest X-ray (Imaging)", "Hybrid (Late Fusion)"])

with tab1:
    st.subheader('Clinical Form')
    cols = st.columns(2)

    with cols[0]:
        age = st.number_input('age', 1.0, 120.0, 57.0)
        trestbps = st.number_input('trestbps', 0.0, 300.0, 130.0)
        chol = st.number_input('chol', 0.0, 800.0, 236.0)
        thalach = st.number_input('thalach', 0.0, 300.0, 174.0)
        oldpeak = st.number_input('oldpeak', 0.0, 10.0, 0.0)

    with cols[1]:
        sex = st.selectbox('sex', [1, 0], index=0)
        cp = st.selectbox('cp', [0, 1, 2, 3], index=2)
        fbs = st.selectbox('fbs', [0, 1], index=0)
        restecg = st.selectbox('restecg', [0, 1, 2], index=1)
        exang = st.selectbox('exang', [0, 1], index=0)
        slope = st.selectbox('slope', [0, 1, 2], index=1)
        ca = st.selectbox('ca', [0, 1, 2, 3, 4], index=0)
        thal = st.selectbox('thal', [0, 1, 2, 3], index=2)

    if st.button('Predict (Tabular)'):
        payload = dict(
            age=age, trestbps=trestbps, chol=chol, thalach=thalach, oldpeak=oldpeak,
            sex=sex, cp=cp, fbs=fbs, restecg=restecg, exang=exang, slope=slope, ca=ca, thal=thal
        )
        res = predict_tabular(payload)  # expects {'prediction': int, 'probability': float}
        st.metric('Prediction (1=disease)', res['prediction'])
        st.metric('Probability', f"{res['probability']:.3f}")

with tab2:
    st.subheader('Upload Chest X-ray')
    f = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'], key="img_xray")

    if st.button('Predict (Imaging)'):
        if f is None:
            st.error("Please upload an image first.")
        else:
            img = Image.open(f).convert('RGB')
            st.image(img, caption="Uploaded X-ray", use_container_width=True)

            model, device, img_size = get_cnn()
            tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
            x = tf(img).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                logit, _ = model(x)   # assuming your model returns (logit, features)
                prob = torch.sigmoid(logit).item()

            st.metric('Prediction (1=Cardiomegaly)', int(prob >= 0.5))
            st.metric('Probability', f"{prob:.3f}")

with tab3:
    st.subheader("Hybrid (Late Fusion)")
    img_file = st.file_uploader("Upload image for hybrid", type=["jpg", "jpeg", "png"], key="hyb_up")
    row_index = st.number_input("UCI row index", min_value=0, step=1, value=0)
    alpha = st.slider("Alpha (weight on imaging)", 0.0, 1.0, 0.5, 0.05)

    if st.button("Predict (Hybrid)"):
        # Validate inputs
        if img_file is None:
            st.error("Upload an image first")
        else:
            import pandas as pd

            # Imaging prob
            model, device, img_size = get_cnn()
            tf = transforms.Compose([transforms.Resize((img_size, img_size)),
                                     transforms.ToTensor()])
            img = Image.open(img_file).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                logit, _ = model(x)
                p_img = float(torch.sigmoid(logit).item())

            # Tabular prob (by row from local UCI copy)
            df = load_uci()
            if int(row_index) >= len(df):
                st.error("Row index out of range")
                st.stop()

            row = df.iloc[int(row_index)].drop(labels=["target"], errors="ignore")
            X = pd.DataFrame([row])
            tab = load_tabular_model()
            p_tab = float(tab.predict_proba(X)[0, 1])

            # Late fusion
            p_h = alpha * p_img + (1 - alpha) * p_tab
            pred = int(p_h >= 0.5)

            st.json({
                "alpha": alpha,
                "p_image": round(p_img, 6),
                "p_tabular": round(p_tab, 6),
                "p_hybrid": round(p_h, 6),
                "prediction": pred
            })
















