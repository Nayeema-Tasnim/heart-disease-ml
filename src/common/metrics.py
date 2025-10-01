from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import numpy as np
def binary_metrics(y_true,y_prob,threshold:float=0.5):
 import numpy as np
 y_pred=(np.array(y_prob)>=threshold).astype(int)
 out={'accuracy':float(accuracy_score(y_true,y_pred)),'precision':float(precision_score(y_true,y_pred,zero_division=0)),'recall':float(recall_score(y_true,y_pred,zero_division=0)),'f1':float(f1_score(y_true,y_pred,zero_division=0))}
 try: out['roc_auc']=float(roc_auc_score(y_true,y_prob))
 except Exception: out['roc_auc']=None
 return out
