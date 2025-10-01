import argparse, json, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.imaging.dataset import CXRDataset
from src.imaging.model import ResNetBinary
from src.common.paths import IMAGING_MODELS
from src.common.metrics import binary_metrics
def main(a):
    device=torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    tf_tr=transforms.Compose([transforms.Resize((a.img_size,a.img_size)),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    tf_val=transforms.Compose([transforms.Resize((a.img_size,a.img_size)),transforms.ToTensor()])
    ds=CXRDataset(a.data_dir,transform=tf_tr,subset=a.subset); n=len(ds); n_val=max(1,int(0.2*n)); n_tr=n-n_val
    tr,va=random_split(ds,[n_tr,n_val])
    tr.dataset.transform=tf_tr; va.dataset.transform=tf_val
    dl_tr=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers)
    dl_va=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers)
    m=ResNetBinary(pretrained=a.pretrained).to(device); opt=torch.optim.AdamW(m.parameters(),lr=a.lr); crit=nn.BCEWithLogitsLoss()
    best_auc=-1.0; IMAGING_MODELS.mkdir(parents=True,exist_ok=True); best_path=IMAGING_MODELS/'best_cnn.pt'
    for epoch in range(1,a.epochs+1):
        m.train(); 
        for x,y,_ in dl_tr:
            x=x.to(device); y=y.float().to(device); logit,_=m(x); loss=crit(logit,y); opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); import numpy as np; P=[]; Y=[]; 
        with torch.no_grad():
            for x,y,_ in dl_va:
                x=x.to(device); y=y.to(device); logit,_=m(x); P.append(torch.sigmoid(logit).cpu().numpy()); Y.append(y.cpu().numpy())
        import numpy as np
        y_prob=np.concatenate(P); y_true=np.concatenate(Y).astype(int); mets=binary_metrics(y_true,y_prob)
        if mets['roc_auc'] is not None and mets['roc_auc']>best_auc: best_auc=mets['roc_auc']; torch.save({'state_dict':m.state_dict(),'arch':'resnet18','img_size':a.img_size}, best_path)
        print('Epoch',epoch,'val',mets)
    (IMAGING_MODELS/'metrics_imaging.json').write_text(json.dumps({'best_val_auc':best_auc},indent=2)); print('Saved',best_path)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir',required=True); ap.add_argument('--subset',default=None)
    ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--batch_size',type=int,default=32); ap.add_argument('--img_size',type=int,default=224)
    ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--num_workers',type=int,default=0); ap.add_argument('--pretrained',type=lambda s: s.lower() in ['true','1','yes'],default=False); ap.add_argument('--cpu',action='store_true')
    main(ap.parse_args())