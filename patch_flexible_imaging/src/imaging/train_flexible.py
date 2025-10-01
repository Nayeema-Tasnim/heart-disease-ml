import argparse, json, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np

from src.imaging.dataset import CXRDataset
from src.imaging.model import ResNetBinary
from src.common.paths import IMAGING_MODELS
from src.common.metrics import binary_metrics

from src.imaging.dataset_csv import CSVSpec, CXRCsvDataset

def loaders_folder(data_dir, subset, img_size, batch_size, num_workers, seed):
    tf_tr=transforms.Compose([transforms.Resize((img_size,img_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    tf_va=transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
    ds=CXRDataset(data_dir, transform=tf_tr, subset=subset); n=len(ds); n_va=max(1,int(0.2*n)); n_tr=n-n_va
    tr,va=random_split(ds,[n_tr,n_va], generator=torch.Generator().manual_seed(seed)); va.dataset.transform=tf_va
    return DataLoader(tr,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True), DataLoader(va,batch_size=batch_size,shuffle=False,num_workers=num_workers)

def loaders_csv(csv, image_root, path_col, label_col, split_col, use_split, pos_labels, neg_labels, img_size, batch_size, num_workers):
    tf_tr=transforms.Compose([transforms.Resize((img_size,img_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    tf_va=transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
    spec=CSVSpec(csv_path=csv, image_root=image_root or '.', path_col=path_col, label_col=label_col, split_col=split_col, use_split=use_split, pos_labels=pos_labels, neg_labels=neg_labels)
    ds=CXRCsvDataset(spec, transform=tf_tr)
    if use_split is None:
        n=len(ds); n_va=max(1,int(0.2*n)); n_tr=n-n_va
        tr,va=random_split(ds,[n_tr,n_va]); va.dataset.transform=tf_va
    else:
        other='val' if str(use_split).lower()=='train' else 'train'
        spec_val=CSVSpec(csv_path=csv, image_root=image_root or '.', path_col=path_col, label_col=label_col, split_col=split_col, use_split=other, pos_labels=pos_labels, neg_labels=neg_labels)
        tr=CXRCsvDataset(spec, transform=tf_tr); va=CXRCsvDataset(spec_val, transform=tf_va)
    return DataLoader(tr,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True), DataLoader(va,batch_size=batch_size,shuffle=False,num_workers=num_workers)

def main(a):
    device=torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    if a.csv:
        dl_tr, dl_va = loaders_csv(a.csv,a.image_root,a.path_col,a.label_col,a.split_col,a.use_split,a.pos_labels,a.neg_labels,a.img_size,a.batch_size,a.num_workers)
    else:
        dl_tr, dl_va = loaders_folder(a.data_dir,a.subset,a.img_size,a.batch_size,a.num_workers,a.seed)
    m=ResNetBinary(pretrained=a.pretrained).to(device); opt=torch.optim.AdamW(m.parameters(),lr=a.lr); crit=nn.BCEWithLogitsLoss()
    best_auc=-1.0; (IMAGING_MODELS).mkdir(parents=True,exist_ok=True); best_path=IMAGING_MODELS/'best_cnn.pt'
    for epoch in range(1,a.epochs+1):
        m.train()
        for x,y,_ in dl_tr:
            x=x.to(device); y=y.float().to(device); logit,_=m(x); loss=crit(logit,y); opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); P=[]; Y=[]
        with torch.no_grad():
            for x,y,_ in dl_va:
                x=x.to(device); y=y.to(device); from torch import sigmoid; P.append(sigmoid(m(x)[0]).cpu().numpy()); Y.append(y.cpu().numpy())
        import numpy as np
        y_prob=np.concatenate(P); y_true=np.concatenate(Y).astype(int); mets=binary_metrics(y_true,y_prob)
        if mets.get('roc_auc') is not None and mets['roc_auc']>best_auc: best_auc=mets['roc_auc']; torch.save({'state_dict':m.state_dict(),'arch':'resnet18','img_size':a.img_size}, best_path)
        print(f'Epoch {epoch}  val: {mets}')
    (IMAGING_MODELS/'metrics_imaging.json').write_text(json.dumps({'best_val_auc':best_auc},indent=2)); print('Saved', best_path)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=None); ap.add_argument('--subset', default=None)
    ap.add_argument('--csv', default=None); ap.add_argument('--image_root', default=None); ap.add_argument('--path_col', default=None); ap.add_argument('--label_col', default=None); ap.add_argument('--split_col', default=None); ap.add_argument('--use_split', default=None)
    ap.add_argument('--pos_labels', nargs='*', default=None); ap.add_argument('--neg_labels', nargs='*', default=None)
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--batch_size', type=int, default=32); ap.add_argument('--img_size', type=int, default=224); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--pretrained', type=lambda s: s.lower() in ['true','1','yes'], default=False); ap.add_argument('--cpu', action='store_true'); ap.add_argument('--seed', type=int, default=42)
    a=ap.parse_args()
    if not a.csv and not a.data_dir: ap.error('Provide either --csv or --data_dir.')
    main(a)
