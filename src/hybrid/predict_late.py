import json
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import joblib

from src.imaging.infer import load_model as load_cnn
from src.common.paths import HYBRID_MODELS, TABULAR_MODELS

def predict_late(image_path: str, uci_csv: str, row_index: int, alpha: float | None = None):
    # 1) Imaging probability
    model, device, img_size = load_cnn()
    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit, _ = model(x)
        p_img = torch.sigmoid(logit).item()

    # 2) Tabular probability
    df = pd.read_csv(uci_csv)
    row = df.iloc[int(row_index)].drop(labels=["target"], errors="ignore")
    X = pd.DataFrame([row])
    tab_model = joblib.load(TABULAR_MODELS / "best_model.joblib")
    p_tab = float(tab_model.predict_proba(X)[0, 1])

    # 3) Load alpha (late-fusion weight) or default to 0.5
    if alpha is None:
        try:
            meta = json.loads((HYBRID_MODELS / "alpha.json").read_text())
            alpha = float(meta.get("alpha", 0.5))
        except Exception:
            alpha = 0.5

    p_hybrid = alpha * p_img + (1 - alpha) * p_tab
    pred = int(p_hybrid >= 0.5)

    return {
        "alpha": float(alpha),
        "p_image": float(p_img),
        "p_tabular": float(p_tab),
        "p_hybrid": float(p_hybrid),
        "prediction": pred  # 1 = positive, 0 = negative
    }

if __name__ == "__main__":
    import argparse, json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--uci_csv", required=True)
    ap.add_argument("--row", type=int, required=True, help="Row index in UCI CSV")
    ap.add_argument("--alpha", type=float, default=None, help="Override alpha if you want")
    args = ap.parse_args()
    out = predict_late(args.image, args.uci_csv, args.row, args.alpha)
    print(_json.dumps(out, indent=2))
