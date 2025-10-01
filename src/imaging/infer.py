from PIL import Image
import torch
from torchvision import transforms
from src.imaging.model import ResNetBinary
from src.common.paths import IMAGING_MODELS

def load_model(path=None, device=None, img_size=224):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = path or (IMAGING_MODELS / "best_cnn.pt")
    ckpt = torch.load(path, map_location=device)
    model = ResNetBinary(arch=ckpt.get("arch", "resnet18"))
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(device)
    return model, device, ckpt.get("img_size", img_size)

def predict_image(image_path: str, model_path: str | None = None):
    model, device, img_size = load_model(model_path)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit, _ = model(x)
        prob = torch.sigmoid(logit).item()
    return {"probability": float(prob), "prediction": int(prob >= 0.5)}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model_path", default=None)
    args = ap.parse_args()
    out = predict_image(args.image, args.model_path)
    print(out)
