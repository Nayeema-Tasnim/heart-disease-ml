import argparse, csv
from pathlib import Path
def main(a):
    root=Path(a.root); rows=[]
    for p in root.rglob('*'):
        if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']:
            rows.append({'image_path': str(p.relative_to(root)).replace('\\','/'), 'label': p.parent.name})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['image_path','label']); w.writeheader(); w.writerows(rows)
    print('Wrote', a.out, 'rows:', len(rows))
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    main(ap.parse_args())
