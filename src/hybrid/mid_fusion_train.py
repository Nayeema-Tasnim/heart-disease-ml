import argparse, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.imaging.model import ResNetBinary
from src.common.paths import HYBRID_MODELS
from src.common.metrics import binary_metrics

TAB_FEATURES=['age','trestbps','chol','thalach','oldpeak','sex','cp','fbs','restecg','exang','slope','ca','thal']

class PairedDS(Dataset):
    def __init__(self, mapping_csv, uci_csv, image_root, img_size=224):
        self.map=pd.read_csv(mapping_csv); self.uci=pd.read_csv(uci_csv); self.root=Path(image_root)
        self.tf=transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
    def __len__(self): return len(self.map)
    def __getitem__(self,i):
        r=self.map.iloc[i]; img=Image.open(self.root/r['image_path']).convert('RGB'); x=self.tf(img)
        row=self.uci.iloc[int(r['uci_row'])].drop(labels=['target'],errors='ignore').reindex(TAB_FEATURES).fillna(0).astype(float).values.astype('float32')
        y=int(r.get('label', 1 if 'Cardiomegaly' in str(r['image_path']) else 0))
        return x, torch.tensor(row), y

class MidFusion(nn.Module):
    def __init__(self, tab_in=13, tab_hidden=32):
        super().__init__()
        self.cnn=ResNetBinary(); self.cnn.head=nn.Identity(); in_f=self.cnn.backbone.fc.in_features
        self.tab=nn.Sequential(nn.Linear(tab_in,tab_hidden), nn.ReLU(inplace=True), nn.BatchNorm1d(tab_hidden))
        self.fusion=nn.Sequential(nn.Linear(in_f+tab_hidden,64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64,1))
    def forward(self,img,tab): 
        feat_img=self.cnn.backbone(img); feat_tab=self.tab(tab); z=torch.cat([feat_img,feat_tab],1); return self.fusion(z).squeeze(1)

def main(a):
    device=torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    ds=PairedDS(a.mapping, a.uci_csv, a.image_root, img_size=a.img_size); n=len(ds); n_val=max(1,int(0.2*n)); n_tr=n-n_val
    tr,va=random_split(ds,[n_tr,n_val])
    dl_tr=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers); dl_va=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers)
    m=MidFusion().to(device); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4); crit=nn.BCEWithLogitsLoss(); best_auc=-1.0
    for epoch in range(1,a.epochs+1):
        m.train()
        for img,tab,y in dl_tr:
            img,tab,y=img.to(device),tab.to(device),y.float().to(device)
            logit=m(img,tab); loss=crit(logit,y); opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); import numpy as np; P=[]; Y=[]
        with torch.no_grad():
            for img,tab,y in dl_va:
                img,tab,y=img.to(device),tab.to(device),y.to(device)
                P.append(torch.sigmoid(m(img,tab)).cpu().numpy()); Y.append(y.cpu().numpy())
        y_prob=np.concatenate(P); y_true=np.concatenate(Y).astype(int); mets=binary_metrics(y_true,y_prob)
        if mets['roc_auc'] is not None and mets['roc_auc']>best_auc: best_auc=mets['roc_auc']; HYBRID_MODELS.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':m.state_dict(),'img_size':a.img_size}, HYBRID_MODELS/'mid_fusion.pt')
        print('Epoch',epoch,'val',mets)
    (HYBRID_MODELS/'metrics_hybrid_mid.json').write_text(json.dumps({'best_val_auc':best_auc},indent=2)); print('Saved mid-fusion.")
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mapping',required=True); ap.add_argument('--uci_csv',required=True); ap.add_argument('--image_root',required=True); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--batch_size',type=int,default=16); ap.add_argument('--img_size',type=int,default=224); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--num_workers',type=int,default=0); ap.add_argument('--cpu',action='store_true'); main(ap.parse_args()))