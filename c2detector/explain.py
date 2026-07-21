from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import RANDOM_STATE
from .utils import ensure_parent

def shap_matrix(explainer, data):
    values=explainer.shap_values(data)
    if isinstance(values,list): values=values[-1]
    arr=np.asarray(values)
    if arr.ndim==3: arr=arr[:,:,-1]
    return arr

def create_global_shap_plot(model,X_test,output_path:Path,sample_size:int):
    if len(X_test)==0: return []
    sample=X_test.sample(n=min(sample_size,len(X_test)),random_state=RANDOM_STATE); explainer=shap.TreeExplainer(model); values=shap_matrix(explainer,sample)
    shap.summary_plot(values,sample,feature_names=sample.columns.tolist(),show=False); ensure_parent(output_path); plt.savefig(output_path,bbox_inches="tight",dpi=160); plt.close()
    order=np.argsort(np.abs(values).mean(axis=0))[::-1]; return [sample.columns[i] for i in order[:10]]

def top_shap_reasons(feature_names:Sequence[str],feature_values,shap_values,top_n:int):
    order=np.argsort(np.abs(shap_values))[::-1][:top_n]; out=[]
    for idx in order:
        contribution=float(shap_values[idx]); out.append({"feature":str(feature_names[idx]),"value":float(feature_values[idx]),"shap_contribution":contribution,"effect":"toward C2" if contribution>0 else "toward benign"})
    return out
