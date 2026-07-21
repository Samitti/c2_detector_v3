from __future__ import annotations
from typing import Any, Dict
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from .config import RANDOM_STATE

def build_model(y_train: pd.Series) -> xgb.XGBClassifier:
    neg=int((y_train==0).sum()); pos=int((y_train==1).sum())
    if not neg or not pos: raise ValueError("Training data must contain both benign and malicious examples.")
    ratio=neg/pos; print(f"  Training class ratio   : {ratio:.2f} benign per malicious")
    return xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.08,subsample=0.9,colsample_bytree=0.9,min_child_weight=1,reg_lambda=1.0,objective="binary:logistic",eval_metric="logloss",scale_pos_weight=ratio,random_state=RANDOM_STATE,n_jobs=1,tree_method="hist")

def compute_metrics(y_true, y_pred, y_prob) -> Dict[str,Any]:
    cm=confusion_matrix(y_true,y_pred,labels=[0,1]); tn,fp,fn,tp=[int(v) for v in cm.ravel()]
    report=classification_report(y_true,y_pred,labels=[0,1],target_names=["benign","c2_malicious"],output_dict=True,zero_division=0)
    roc=float(roc_auc_score(y_true,y_prob)) if pd.Series(y_true).nunique()==2 else None
    return {"accuracy":float(accuracy_score(y_true,y_pred)),"roc_auc":roc,"false_positive_rate":fp/(fp+tn) if fp+tn else None,"false_negative_rate":fn/(fn+tp) if fn+tp else None,"confusion_matrix":{"tn":tn,"fp":fp,"fn":fn,"tp":tp},"classification_report":report,"support":int(len(y_true))}

def print_metrics(title: str, metrics: Dict[str,Any]):
    print(f"\n  {title}\n  " + "-"*66); r=metrics["classification_report"]
    print(f"  {'CLASS':<20}{'PRECISION':>12}{'RECALL':>10}{'F1':>10}{'SUPPORT':>12}")
    for key,name in [("benign","Benign"),("c2_malicious","C2 Malicious")]:
        row=r[key]; print(f"  {name:<20}{row['precision']:>12.4f}{row['recall']:>10.4f}{row['f1-score']:>10.4f}{int(row['support']):>12,}")
    print(f"\n  Accuracy             : {metrics['accuracy']:.6f}")
    print(f"  ROC-AUC              : {metrics['roc_auc']:.6f}" if metrics['roc_auc'] is not None else "  ROC-AUC              : N/A")
    print(f"  False Positive Rate  : {metrics['false_positive_rate']:.8f}" if metrics['false_positive_rate'] is not None else "  False Positive Rate  : N/A")
    print(f"  False Negative Rate  : {metrics['false_negative_rate']:.8f}" if metrics['false_negative_rate'] is not None else "  False Negative Rate  : N/A")
