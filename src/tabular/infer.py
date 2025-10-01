import pandas as pd, joblib
from src.common.paths import TABULAR_MODELS
NUMERIC=['age','trestbps','chol','thalach','oldpeak']; CATEG=['sex','cp','fbs','restecg','exang','slope','ca','thal']
def predict_one(payload:dict, model_path=None):
 model_path=model_path or (TABULAR_MODELS/'best_model.joblib'); m=joblib.load(model_path); import pandas as pd
 df=pd.DataFrame([payload],columns=NUMERIC+CATEG); p=float(m.predict_proba(df)[0,1]); return {'prediction':int(p>=0.5),'probability':p}
