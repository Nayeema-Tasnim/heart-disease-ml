from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

COMMON_PATH_COLS=["image_path","path","filepath","file","image","img"]
COMMON_LABEL_COLS=["label","class","finding","target","y"]

def _auto_pick(cols, cands):
    low=[c.lower() for c in cols]
    for cand in cands:
        if cand in low: return cols[low.index(cand)]
    return None

def _norm(x): return ("" if x is None else str(x)).strip()

@dataclass
class CSVSpec:
    csv_path: str
    image_root: str = "."
    path_col: Optional[str] = None
    label_col: Optional[str] = None
    split_col: Optional[str] = None
    use_split: Optional[str] = None
    pos_labels: Optional[List[str]] = None
    neg_labels: Optional[List[str]] = None

class CXRCsvDataset(Dataset):
    def __init__(self, spec: CSVSpec, transform=None):
        self.spec=spec; self.transform=transform; self.root=Path(spec.image_root)
        df=pd.read_csv(spec.csv_path)
        if spec.split_col and spec.use_split:
            df=df[df[spec.split_col].astype(str).str.lower()==str(spec.use_split).lower()]
        path_col=spec.path_col or _auto_pick(df.columns.tolist(), COMMON_PATH_COLS)
        label_col=spec.label_col or _auto_pick(df.columns.tolist(), COMMON_LABEL_COLS)
        if path_col is None or label_col is None:
            raise ValueError("Could not detect path/label columns. Use --path_col/--label_col.")
        self.path_col, self.label_col = path_col, label_col
        pos=set([_norm(x).lower() for x in (spec.pos_labels or ['cardiomegaly','cm'])])
        neg=set([_norm(x).lower() for x in (spec.neg_labels or ['no finding','normal','none'])])
        items=[]
        for _,r in df.iterrows():
            rel=str(r[path_col]).strip(); lab=_norm(r[label_col]).lower()
            if lab in pos: y=1
            elif lab in neg: y=0
            else:
                try:
                    y=int(float(lab)); 
                    if y not in (0,1): continue
                except Exception:
                    continue
            items.append((rel,y))
        if not items: raise RuntimeError("No usable rows after mapping labels; check --pos_labels/--neg_labels.")
        self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,i:int):
        rel,y=self.items[i]; from PIL import Image
        img=Image.open(self.root/rel).convert('RGB'); 
        if self.transform: img=self.transform(img)
        return img, y, rel.replace('\\','/')
