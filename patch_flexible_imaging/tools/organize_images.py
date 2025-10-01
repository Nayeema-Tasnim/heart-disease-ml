import argparse, shutil
from pathlib import Path
import pandas as pd
def main(a):
    df=pd.read_csv(a.csv); src=Path(a.src_root); dst=Path(a.dst_root)
    pos=set(s.lower() for s in a.pos_labels); neg=set(s.lower() for s in a.neg_labels)
    moved=0
    for _,r in df.iterrows():
        p = Path(r[a.path_col])
        if not p.is_absolute(): p = src / p
        if not p.exists(): print('Missing:', p); continue
        lab = str(r[a.label_col]).strip().lower()
        if lab in pos: sub='Cardiomegaly'
        elif lab in neg: sub='No Finding'
        else: continue
        d = dst / sub / p.name; d.parent.mkdir(parents=True, exist_ok=True)
        if a.copy: shutil.copy2(p, d)
        else: shutil.move(str(p), str(d))
        moved+=1
    print(('Copied' if a.copy else 'Moved'), moved, 'images to', dst)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv', required=True); ap.add_argument('--path_col', default='image_path'); ap.add_argument('--label_col', default='label')
    ap.add_argument('--src_root', required=True); ap.add_argument('--dst_root', required=True)
    ap.add_argument('--pos_labels', nargs='*', default=['Cardiomegaly','CM']); ap.add_argument('--neg_labels', nargs='*', default=['No Finding','Normal','None'])
    ap.add_argument('--copy', action='store_true')
    main(ap.parse_args())
