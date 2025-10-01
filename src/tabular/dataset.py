import pandas as pd, numpy as np
def load_uci_csv(path:str)->pd.DataFrame:
 df=pd.read_csv(path); df.columns=[c.strip().lower() for c in df.columns]; df=df.replace('?',np.nan)
 if 'target' not in df.columns and 'num' in df.columns: df=df.rename(columns={'num':'target'})
 df['target']=pd.to_numeric(df['target'],errors='coerce').fillna(0).astype(int); df['target']=(df['target']>0).astype(int)
 return df
