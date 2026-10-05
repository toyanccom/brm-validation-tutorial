#!/usr/bin/env python3
from pathlib import Path
import argparse,pandas as pd,numpy as np
from sklearn.model_selection import KFold,GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from mot_repro_utils import SEED,metric

def aggregate(root,e,f):
 d=pd.read_excel(Path(root)/f'Exp{e}'/f'exp{e}_{f}_merged.xlsx')
 if e==1: gc=['mod_type','mod_field','Gender','Age','participant'];cats=['mod_type','mod_field','Gender'];nums=['Age']
 else: gc=['mod_type','Gender','Age','participant'];cats=['mod_type','Gender'];nums=['Age']
 return d.groupby(gc,dropna=False)['corr_clicks'].mean().reset_index(name='y'),cats,nums

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--out',default='results');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
 for e in [1,2]:
  for f in ['blur','contrast']:
   d,cats,nums=aggregate(a.data_root,e,f);X=d[cats+nums].reset_index(drop=True);y=d.y.to_numpy();g=d.participant.astype(str).to_numpy()
   for scheme in ['rowwise','participant_grouped']:
    pred=np.zeros(len(y));ovs=[];splits=KFold(5,shuffle=True,random_state=SEED).split(X,y) if scheme=='rowwise' else GroupKFold(5).split(X,y,g)
    for fi,(tr,te) in enumerate(splits):
     pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cats)],remainder='passthrough');model=Pipeline([('pre',pre),('rf',RandomForestRegressor(n_estimators=100,random_state=SEED+fi,n_jobs=-1))]);model.fit(X.iloc[tr],y[tr]);pred[te]=model.predict(X.iloc[te]);ovs.append(len(set(g[tr])&set(g[te]))/len(set(g[te])))
    mt=metric(y,pred);rows.append([e,f,scheme,mt['MAE'],mt['RMSE'],mt['R2'],float(np.mean(ovs))])
 pd.DataFrame(rows,columns=['experiment','feature','validation','MAE','RMSE','R2','mean_test_participant_overlap']).to_csv(out/'legacy_validation_comparison.csv',index=False)
if __name__=='__main__':main()
