import argparse, random, pandas as pd
from pathlib import Path
from src.imaging.dataset import find_images

def main(a):
 rng=random.Random(a.seed); ims=find_images(Path(a.image_root),subset=a.subset); df=pd.read_csv(a.uci_csv)
 n=min(a.n if a.n>0 else len(ims), len(df)); rows=[]; used=set()
 for _ in range(n):
  rel,label=ims[rng.randrange(0,len(ims))]
  while True:
   u=rng.randrange(0,len(df))
   if u not in used: used.add(u); break
  rows.append({'image_path':rel,'uci_row':u,'label':label})
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); print('Wrote',out)
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--uci_csv',required=True); ap.add_argument('--image_root',required=True); ap.add_argument('--subset',default=None); ap.add_argument('--out',required=True); ap.add_argument('--n',type=int,default=500); ap.add_argument('--seed',type=int,default=42); main(ap.parse_args())
