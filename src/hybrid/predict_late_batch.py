import json
import pandas as pd
from pathlib import Path
from src.hybrid.predict_late import predict_late

def main(mapping="data/pairs/id_map.csv", uci_csv="data/uci/heart_cleveland.csv", image_root="data/cxr", alpha=None, out="data/pairs/preds_late.csv"):
    mp = pd.read_csv(mapping)
    rows = []
    for _, r in mp.iterrows():
        image_path = str(Path(image_root) / r["image_path"])
        res = predict_late(image_path=image_path, uci_csv=uci_csv, row_index=int(r["uci_row"]), alpha=alpha)
        rows.append({**r, **res})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} with {len(rows)} rows")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="data/pairs/id_map.csv")
    ap.add_argument("--uci_csv", default="data/uci/heart_cleveland.csv")
    ap.add_argument("--image_root", default="data/cxr")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--out", default="data/pairs/preds_late.csv")
    a = ap.parse_args()
    main(a.mapping, a.uci_csv, a.image_root, a.alpha, a.out)
