from pathlib import Path
import os, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
SEED=42
IMG=['target1_img','target2_img','target3_img','target4_img','distractor1_img','distractor2_img','distractor3_img','distractor4_img']

def segment_runs(d):
    d=d.copy().reset_index(drop=True); run=np.zeros(len(d),dtype=int)
    for p,inds in d.groupby('participant',sort=False).indices.items():
        r=0; pn=None; pdte=None
        for j in inds:
            n=int(d.at[j,'trials.thisN']); dt=str(d.at[j,'date'])
            if pn is not None and (n<=pn or dt!=pdte): r+=1
            run[j]=r; pn=n; pdte=dt
    d['run_index']=run; return d

def complete_runs(d,exp):
    expected=list(range(80 if exp==1 else 48)); out=[]
    for (p,r),g in d.groupby(['participant','run_index'],sort=False):
        if sorted(pd.to_numeric(g['trials.thisN']).astype(int).unique())==expected: out.append((p,r))
    return out

def choose_run(d,exp,which='first'):
    ch={}
    for p,r in complete_runs(d,exp):
        if which=='first': ch.setdefault(p,r)
        else: ch[p]=r
    return d[np.array([ch.get(p)==r for p,r in zip(d.participant,d.run_index)])].copy()

def build_frame(root,exp,feat,which='first',only4=False,exclude_nmods5=False,baseline_first_trial=False):
    d=choose_run(segment_runs(pd.read_excel(Path(root)/f'Exp{exp}'/f'exp{exp}_{feat}_merged.xlsx')),exp,which)
    d['heterogeneity']=d[IMG].apply(lambda r:str(len(set(os.path.basename(str(v)) for v in r))),axis=1)
    if exp==1 and baseline_first_trial:
        b=d[pd.to_numeric(d['trials.thisN']).astype(int)==0].set_index('participant')['corr_clicks']; d['baseline_score']=d.participant.map(b)
    if exp==1:
        n=pd.to_numeric(d['trials.thisN']).astype(int); d=d[n%4!=0].copy(); n=pd.to_numeric(d['trials.thisN']).astype(int); d['condition']=(n//16).astype(str)
    else:
        if exclude_nmods5: d=d[pd.to_numeric(d['n_mods'])!=5].copy()
        d['condition']=pd.to_numeric(d['n_mods']).astype(int).astype(str)
    if only4: d=d[pd.to_numeric(d.total_clicks)==4].copy()
    d['field']=d.mod_field.astype(str); d['gender']=d.Gender.astype(str).str.strip(); d['age']=pd.to_numeric(d.Age); d['framerate']=pd.to_numeric(d.frameRate); d['y']=pd.to_numeric(d.corr_clicks); d['group']=d.participant.astype(str)
    return d.reset_index(drop=True)

def cols(name='full',baseline=False):
    maps={
      'conditions_only':(['condition','field','heterogeneity'],[]),
      'demographics_only':(['gender'],['age']),
      'conditions_demographics':(['condition','field','heterogeneity','gender'],['age']),
      'conditions_technical':(['condition','field','heterogeneity'],['framerate']),
      'full':(['condition','field','heterogeneity','gender'],['age','framerate']),
      'full_minus_age':(['condition','field','heterogeneity','gender'],['framerate']),
      'full_minus_heterogeneity':(['condition','field','gender'],['age','framerate']),
      'full_minus_degradation':(['field','heterogeneity','gender'],['age','framerate'])}
    cats,nums=maps[name]; cats=list(cats); nums=list(nums)
    if baseline: nums.append('baseline_score')
    return cats,nums

def pre(cats,nums):
    tr=[]
    if cats: tr.append(('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cats))
    if nums: tr.append(('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums))
    return ColumnTransformer(tr)

def oof(d,model='Ridge',feature_set='full',baseline=False):
    cats,nums=cols(feature_set,baseline); X=d[cats+nums].reset_index(drop=True); y=d.y.to_numpy(float); g=d.group.to_numpy(); p=np.zeros(len(y)); fitted=[]
    for fi,(tr,te) in enumerate(GroupKFold(5).split(X,y,g)):
        if model=='MeanBaseline': pipe=None; phat=np.repeat(y[tr].mean(),len(te))
        elif model=='Ridge': pipe=Pipeline([('pre',pre(cats,nums)),('model',Ridge(alpha=10.0))]); pipe.fit(X.iloc[tr],y[tr]); phat=pipe.predict(X.iloc[te])
        elif model=='RandomForest': pipe=Pipeline([('pre',pre(cats,nums)),('model',RandomForestRegressor(n_estimators=60,max_depth=3,min_samples_leaf=5,random_state=SEED+fi,n_jobs=-1))]); pipe.fit(X.iloc[tr],y[tr]); phat=pipe.predict(X.iloc[te])
        elif model=='GradientBoosting': pipe=Pipeline([('pre',pre(cats,nums)),('model',GradientBoostingRegressor(n_estimators=80,learning_rate=.03,max_depth=1,min_samples_leaf=8,random_state=SEED+fi))]); pipe.fit(X.iloc[tr],y[tr]); phat=pipe.predict(X.iloc[te])
        elif model=='SVR': pipe=Pipeline([('pre',pre(cats,nums)),('model',SVR(C=.1,kernel='rbf',gamma='scale',epsilon=.1))]); pipe.fit(X.iloc[tr],y[tr]); phat=pipe.predict(X.iloc[te])
        p[te]=phat; fitted.append((tr,te,pipe))
    return dict(y=y,pred=p,X=X,cats=cats,nums=nums,fitted=fitted)

def metric(y,p): return dict(MAE=mean_absolute_error(y,p),RMSE=mean_squared_error(y,p)**.5,R2=r2_score(y,p))
def calibration(y,p):
    m=LinearRegression().fit(p.reshape(-1,1),y); return float(m.intercept_),float(m.coef_[0])
def cluster_ci(d,y,p,B=400,seed=SEED):
    rng=np.random.default_rng(seed); groups=np.array(sorted(pd.unique(d.group.astype(str)))); idx={g:np.flatnonzero(d.group.astype(str).to_numpy()==g) for g in groups}; vals=[]
    for _ in range(B):
        smp=rng.choice(groups,size=len(groups),replace=True); ii=np.concatenate([idx[g] for g in smp]); vals.append(r2_score(y[ii],p[ii]))
    return np.quantile(vals,[.025,.975])
