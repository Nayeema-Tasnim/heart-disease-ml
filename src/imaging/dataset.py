from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
CLASS_TO_IDX={'No Finding':0,'Cardiomegaly':1}

def find_images(root:Path, subset=None):
 root=Path(root); root=root/subset if subset else root; items=[]
 for cls,idx in CLASS_TO_IDX.items():
  p=root/cls
  if not p.exists(): continue
  for q in p.rglob('*'):
   if q.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']:
    items.append((str(q.relative_to(root)).replace('\\','/'), idx))
 return items

class CXRDataset(Dataset):
 def __init__(self, root, transform=None, subset=None, filelist=None):
  from pathlib import Path
  self.root=Path(root); self.transform=transform
  self.items=filelist if filelist is not None else find_images(self.root, subset)
 def __len__(self): return len(self.items)
 def __getitem__(self,i):
  rel,label=self.items[i]; from PIL import Image
  img=Image.open(self.root/rel).convert('RGB')
  if self.transform: img=self.transform(img)
  return img,label,rel
