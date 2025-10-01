from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import io, torch
from torchvision import transforms
from src.tabular.infer import predict_one as predict_tabular
from src.imaging.infer import load_model as load_cnn
app=FastAPI(title='Heart Disease Multimodal API',version='1.0.0')
class Patient(BaseModel):
 age:float; trestbps:float; chol:float; thalach:float; oldpeak:float; sex:int; cp:int; fbs:int; restecg:int; exang:int; slope:int; ca:int; thal:int
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/predict/tabular')
def p_tab(p:Patient): return predict_tabular(p.model_dump())
@app.post('/predict/image')
async def p_img(file:UploadFile=File(...)):
 m,device,img_size=load_cnn(); tf=transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()])
 img=Image.open(io.BytesIO(await file.read())).convert('RGB'); x=tf(img).unsqueeze(0).to(device)
 with torch.no_grad(): logit,_=m(x); import torch as T; prob=T.sigmoid(logit).item()
 return {'prediction':int(prob>=0.5),'probability':prob}
