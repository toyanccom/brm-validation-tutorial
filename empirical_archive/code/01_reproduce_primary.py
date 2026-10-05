#!/usr/bin/env python3
from pathlib import Path
import argparse, pandas as pd, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from mot_repro_utils import *
from mot_common import provenance
MODELS=['MeanBaseline','Ridge','RandomForest','GradientBoosting','SVR']
SETS=['conditions_only','demographics_only','conditions_demographics','conditions_technical','full','full_minus_age','full_minus_heterogeneity','full_minus_degradation']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--out',default='results');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 provenance(a.data_root,out)
 frames={(e,f):build_frame(a.data_root,e,f) for e in [1,2] for f in ['blur','contrast']}
 main=[]; cache={}
 for (e,f),d in frames.items():
  for mi,m in enumerate(MODELS):
   r=oof(d,m);mt=metric(r['y'],r['pred']);ci=cluster_ci(d,r['y'],r['pred'],400,SEED+mi);main.append([e,f,m,mt['MAE'],mt['RMSE'],mt['R2'],ci[0],ci[1]]);cache[(e,f,m)]=r
 pd.DataFrame(main,columns=['experiment','feature','model','MAE','RMSE','R2','R2_CI_low','R2_CI_high']).to_csv(out/'primary_model_performance.csv',index=False)
 abl=[]
 for (e,f),d in frames.items():
  for fs in SETS:
   r=cache[(e,f,'Ridge')] if fs=='full' else oof(d,'Ridge',fs);mt=metric(r['y'],r['pred']);inter,slope=calibration(r['y'],r['pred']);abl.append([e,f,fs,mt['MAE'],mt['RMSE'],mt['R2'],inter,slope])
 pd.DataFrame(abl,columns=['experiment','feature','feature_set','MAE','RMSE','R2','calibration_intercept','calibration_slope']).to_csv(out/'ridge_feature_sets.csv',index=False)
 sens=[]
 for e in [1,2]:
  for f in ['blur','contrast']:
   for label,kw in [('four_response_only',dict(only4=True)),('last_valid_complete_run',dict(which='last'))]:
    d=build_frame(a.data_root,e,f,**kw);r=oof(d,'Ridge');mt=metric(r['y'],r['pred']);sens.append([e,f,label,mt['MAE'],mt['RMSE'],mt['R2']])
 for f in ['blur','contrast']:
  d=build_frame(a.data_root,2,f,exclude_nmods5=True);r=oof(d,'Ridge');mt=metric(r['y'],r['pred']);sens.append([2,f,'exclude_nmods5',mt['MAE'],mt['RMSE'],mt['R2']])
  d=build_frame(a.data_root,1,f,baseline_first_trial=True);r=oof(d,'Ridge',baseline=True);mt=metric(r['y'],r['pred']);sens.append([1,f,'plus_first_session_baseline_trial',mt['MAE'],mt['RMSE'],mt['R2']])
 pd.DataFrame(sens,columns=['experiment','feature','analysis','MAE','RMSE','R2']).to_csv(out/'sensitivity_analyses.csv',index=False)
 # permutation importance
 rec=[];rng=np.random.default_rng(SEED)
 for (e,f),d in frames.items():
  r=cache[(e,f,'Ridge')];y=r['y'];X=r['X']
  for fi,(tr,te,model) in enumerate(r['fitted']):
   base=mean_absolute_error(y[te],model.predict(X.iloc[te]))
   for feature_index,feat in enumerate(r['cats']+r['nums']):
    for rr in range(8):
     # Separate deterministic stream per dataset/fold/feature/repetition.
     rng=np.random.default_rng(np.random.SeedSequence([SEED,e,0 if f=='blur' else 1,fi,feature_index,rr]))
     xp=X.iloc[te].copy();v=xp[feat].to_numpy().copy();rng.shuffle(v);xp[feat]=v;rec.append([e,f,fi,feat,rr,mean_absolute_error(y[te],model.predict(xp))-base])
 pd.DataFrame(rec,columns=['experiment','feature_family','fold','feature','repeat','delta_MAE']).to_csv(out/'ridge_permutation_importance.csv',index=False)
 # simultaneous participant + condition-level holdout stress
 stress=[]
 for (e,f),d in frames.items():
  cats,nums=cols('full');X=d[cats+nums].reset_index(drop=True);y=d.y.to_numpy();g=d.group.to_numpy();cond=d.condition.to_numpy()
  for lev in sorted(pd.unique(cond),key=lambda x:float(x)):
   pp=[];oo=[]
   for tr0,te0 in GroupKFold(5).split(X,y,g):
    tr=tr0[cond[tr0]!=lev];te=te0[cond[te0]==lev]
    if not len(te):continue
    model=Pipeline([('pre',pre(cats,nums)),('model',Ridge(alpha=10.0))]);model.fit(X.iloc[tr],y[tr]);pp.append(model.predict(X.iloc[te]));oo.append(y[te])
   pp=np.concatenate(pp);oo=np.concatenate(oo);mt=metric(oo,pp);stress.append([e,f,lev,len(oo),mt['MAE'],mt['RMSE'],mt['R2']])
 pd.DataFrame(stress,columns=['experiment','feature_family','held_level','n','MAE','RMSE','R2']).to_csv(out/'leave_level_grouped_stress.csv',index=False)
if __name__=='__main__':main()
