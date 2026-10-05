"""Shared definitions for the 27 September 2026 revision.

All participant identifiers remain internal to the calculation. Public CSV outputs
are aggregate statistics. The source workbooks are checked before any analysis.
"""
from pathlib import Path
import hashlib,json,platform,importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import norm

EXPECTED={
 'Exp1/exp1_blur_merged.xlsx':'b5b9aad91d9c02ba851fab266f91a8d330b6c63ea7a3915e01c4e91cee2ac295',
 'Exp1/exp1_contrast_merged.xlsx':'6314c50df451f40a3be258e12f3849f30fc984b4187cf6509b58c9149b0d1936',
 'Exp2/exp2_blur_merged.xlsx':'0e155f3713b50febd9454c5a7b2734550ac2031d50d4dd00ede3abba07e3726a',
 'Exp2/exp2_contrast_merged.xlsx':'d9a4a7da92264a7b052c071f8be7ceaa5d566133cb49cd3e02a3b7da3453e81b'}
IMAGE_COLS=['target1_img','target2_img','target3_img','target4_img','distractor1_img','distractor2_img','distractor3_img','distractor4_img']
PROFILE={1:'Homogeneous',2:'Two unique',4:'Four unique',8:'Eight unique'}
BOOT_SEED=20260923

def provenance(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for rel,expected in EXPECTED.items():
        p=root/rel;actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Source hash mismatch: {rel}')
        rows.append({'file':rel,'sha256':actual,'status':'MATCH'})
    pd.DataFrame(rows).to_csv(out/'source_verification.csv',index=False)
    packages={}
    for name in ['numpy','pandas','scipy','scikit-learn','statsmodels','patsy','matplotlib','openpyxl']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    (out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'packages':packages,'revision':'2026-09-27'},indent=2))

def segment(raw):
    d=raw.copy().reset_index(drop=True);d['_seg']=0
    for _,g in d.groupby('participant',sort=False):
        n=pd.to_numeric(g['trials.thisN']);dates=g.date.astype(str)
        starts=n.diff().le(0)|dates.ne(dates.shift());starts.iloc[0]=False
        d.loc[g.index,'_seg']=starts.cumsum().to_numpy()
    return d

def load(root,experiment,feature,which='all',dedup=True):
    raw=pd.read_excel(Path(root)/f'Exp{experiment}'/f'exp{experiment}_{feature}_merged.xlsx')
    d=segment(raw.drop_duplicates() if dedup else raw)
    # Preserve literal label types and the original numeric-before-string sort
    # used by the archived bootstrap; coercion to text changes seeded resamples.
    d['label']=d.participant;d['n']=pd.to_numeric(d['trials.thisN']).astype(int)
    expected=list(range(80 if experiment==1 else 48));good=[]
    for (label,seg),g in d.groupby(['label','_seg'],sort=False):
        if sorted(g.n.tolist())==expected:good.append((label,int(seg)))
    if len(good)!=d.groupby(['label','_seg']).ngroups:raise ValueError('Incomplete or repeated-index run detected')
    if which in ['first','last']:
        choice={}
        for label,seg in good:
            if which=='first':choice.setdefault(label,seg)
            else:choice[label]=seg
        d=d[[choice[l]==r for l,r in zip(d.label,d._seg)]].copy()
    d['run']=d[['participant','date','session','expName']].astype(str).agg('|'.join,axis=1)
    d['profile']=d[IMAGE_COLS].apply(lambda row:len(set(str(x).replace('\\','/').split('/')[-1] for x in row)),axis=1)
    d['block']=d.n//16 if experiment==1 else d.n//12
    d['cluster']=d.n//4 if experiment==1 else d.n//3
    d['field']=d.mod_field.astype(str);d['schedule']=(d.n%12)//3
    d['age']=pd.to_numeric(d.Age);d['age10']=d.age/10;d['male']=(d.Gender.astype(str).str.strip().str.lower()=='male').astype(int)
    d['recovery']=pd.to_numeric(d.corr_clicks)/4;d['precision']=pd.to_numeric(d.corr_clicks)/pd.to_numeric(d.total_clicks)
    d['fps']=pd.to_numeric(d.frameRate);d['experiment']=experiment;d['feature']=feature
    vectors=d.all_clicks.astype(str).map(json.loads)
    assert all(len(v)==8 and sum(v)==t and sum(v[:4])==c and sum(v[4:])==i for v,t,c,i in zip(vectors,d.total_clicks,d.corr_clicks,d.incorr_clicks))
    return raw,d.reset_index(drop=True)

def pairs(d,only4=False,outcome='corr_clicks'):
    keys=['label','run','cluster','block','profile']
    b=d[d.n%4==0][keys+[outcome,'total_clicks','age','age10','male','fps']].rename(columns={outcome:'baseline','total_clicks':'base_clicks'})
    m=d[d.n%4!=0][keys+[outcome,'total_clicks','field']].rename(columns={outcome:'manipulated','total_clicks':'mod_clicks'})
    q=m.merge(b,on=keys,validate='many_to_one')
    if only4:q=q[(q.base_clicks==4)&(q.mod_clicks==4)].copy()
    q['delta']=q.manipulated-q.baseline
    return q

def label_means(d,value,by=None):
    by=by or []
    return d.groupby(['label','run']+by,sort=True)[value].mean().groupby(['label']+by,sort=True).mean()

def estimate(values,B=10000,seed=BOOT_SEED):
    a=np.asarray(values,float);a=a[np.isfinite(a)];rng=np.random.default_rng(seed)
    sims=rng.choice(a,size=(B,len(a)),replace=True).mean(axis=1);lo,hi=np.quantile(sims,[.025,.975]);sd=a.std(ddof=1)
    return {'labels':len(a),'mean':a.mean(),'ci_low':lo,'ci_high':hi,'sd':sd,'dz':a.mean()/sd if sd else np.nan}

def summarize(d,value,by):
    rows=[]
    for key,g in d.groupby(by,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        rows.append(dict(zip(by,key))|estimate(label_means(g,value)))
    return pd.DataFrame(rows)

def hc3(y,X):
    y=np.asarray(y,float);X=np.column_stack([np.ones(len(y)),np.asarray(X,float)])
    inv=np.linalg.inv(X.T@X);b=inv@X.T@y;h=np.einsum('ij,jk,ik->i',X,inv,X)
    score=(y-X@b)/(1-h);cov=inv@((X.T*score**2)@X)@inv;se=np.sqrt(np.diag(cov));z=norm.ppf(.975)
    return b,b-z*se,b+z*se

def data_audit(root):
    rows=[]
    for e in [1,2]:
        for f in ['blur','contrast']:
            raw,d=load(root,e,f);_,first=load(root,e,f,'first')
            rawseg=segment(raw);rawreps=rawseg.groupby('participant')._seg.max()
            repeats=d.groupby('label').run.nunique();pm=first.groupby('label').agg(age=('age','first'),male=('male','first'))
            rows.append({'experiment':e,'feature':f,'raw_rows':len(raw),'duplicate_extras':len(raw)-len(raw.drop_duplicates()),'unique_rows':len(d),'labels':d.label.nunique(),'unique_runs':d.run.nunique(),'raw_repeat_identifiers':int((rawreps>0).sum()),'distinct_repeat_identifiers':int((repeats>1).sum()),'first_run_rows':len(first),'non4_unique':int((d.total_clicks!=4).sum()),'age_min':pm.age.min(),'age_max':pm.age.max(),'age_mean':pm.age.mean(),'age_sd':pm.age.std(),'age_median':pm.age.median(),'age_q1':pm.age.quantile(.25),'age_q3':pm.age.quantile(.75),'age_45plus':int((pm.age>=45).sum()),'age_60plus':int((pm.age>=60).sum()),'female':int((pm.male==0).sum()),'male':int(pm.male.sum()),'fps_min':d.fps.min(),'fps_max':d.fps.max(),'vector_mismatches':0,'age_inconsistent_labels':int((d.groupby('label').age.nunique()>1).sum())})
    return pd.DataFrame(rows)
