"""Revision analyses for output derivation, group permutation and split sensitivity.

Requires the primary script output; participant means are written only under
_private for local figure generation and are excluded from the public package.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from mot_repro_utils import *
from mot_common import data_audit,provenance

def calibration_boot(d,y,p):
    rng=np.random.default_rng(42);groups=np.array(sorted(d.group.unique()));g=d.group.to_numpy();ix={v:np.flatnonzero(g==v) for v in groups};vals=[]
    for _ in range(400):
        ii=np.concatenate([ix[v] for v in rng.choice(groups,len(groups),replace=True)]);inter,slope=calibration(y[ii],p[ii]);vals.append([inter,slope])
    return np.array(vals)

def main(root,out):
    out=Path(out);provenance(root,out);private=out/'_private';private.mkdir(exist_ok=True)
    audit=data_audit(root);metrics=[];cal=[];bands=[];perm=[];splits=[];abl=[];plot=[]
    sets=['conditions_only','demographics_only','conditions_demographics','conditions_technical','full','full_minus_age','full_minus_heterogeneity','full_minus_degradation']
    for e in [1,2]:
        for f in ['blur','contrast']:
            d=build_frame(root,e,f);r=oof(d,'Ridge');y=r['y'];p=r['pred'];X=r['X'];groups=d.group.to_numpy()
            audit.loc[(audit.experiment==e)&(audit.feature==f),'modeled_rows']=len(d)
            audit.loc[(audit.experiment==e)&(audit.feature==f),'modeled_non4']=int((d.total_clicks!=4).sum())
            audit.loc[(audit.experiment==e)&(audit.feature==f),'outcome_mean']=y.mean()
            audit.loc[(audit.experiment==e)&(audit.feature==f),'outcome_sd']=y.std(ddof=1)
            pm=pd.DataFrame({'group':groups,'observed':y,'predicted':p}).groupby('group',sort=True)[['observed','predicted']].mean()
            metrics.append({'experiment':e,'feature':f,'target':'participant_mean',**metric(pm.observed,pm.predicted)})
            for _,row in pm.iterrows():plot.append({'experiment':e,'feature':f,'observed':row.observed,'predicted':row.predicted})
            inter,slope=calibration(y,p);bc=calibration_boot(d,y,p);lo,hi=np.quantile(bc,[.025,.975],axis=0)
            cal.append({'experiment':e,'feature':f,'intercept':inter,'intercept_low':lo[0],'intercept_high':hi[0],'slope':slope,'slope_low':lo[1],'slope_high':hi[1]})
            for xx in np.linspace(p.min(),p.max(),40):
                low,high=np.quantile(bc[:,0]+xx*bc[:,1],[.025,.975]);bands.append({'experiment':e,'feature':f,'predicted':xx,'fit':inter+slope*xx,'ci_low':low,'ci_high':high})
            for fs in sets:
                rr=r if fs=='full' else oof(d,'Ridge',fs);mt=metric(rr['y'],rr['pred']);iv,sl=calibration(rr['y'],rr['pred']);ci=cluster_ci(d,rr['y'],rr['pred'],400,42)
                abl.append({'experiment':e,'feature':f,'feature_set':fs,**mt,'R2_CI_low':ci[0],'R2_CI_high':ci[1],'calibration_intercept':iv,'calibration_slope':sl})
            for fi,(_,te,model) in enumerate(r['fitted']):
                base=mean_absolute_error(y[te],model.predict(X.iloc[te]));tg=groups[te];ug=pd.unique(tg)
                for block_index,(name,cols0) in enumerate([('demographic_block',['age','gender']),('framerate',['framerate'])]):
                    person=X.iloc[te].assign(_group=tg).groupby('_group',sort=False)[cols0].first().reindex(ug)
                    for rep in range(8):
                        rng=np.random.default_rng(np.random.SeedSequence([42,99,e,0 if f=='blur' else 1,fi,block_index,rep]));source=rng.permutation(len(ug))
                        donor=pd.DataFrame(person.iloc[source].to_numpy(),index=ug,columns=cols0);xp=X.iloc[te].copy()
                        for col in cols0:xp[col]=pd.Series(tg).map(donor[col]).to_numpy()
                        perm.append({'experiment':e,'feature_family':f,'fold':fi,'feature_block':name,'repeat':rep,'delta_MAE':mean_absolute_error(y[te],model.predict(xp))-base})
            # These are alternative fitted workflows, not selection candidates.
            cats,nums=cols('full');xx=d[cats+nums]
            for seed in [17,29,43,61,89]:
                pred=np.full(len(d),np.nan)
                for tr,te in GroupKFold(5,shuffle=True,random_state=seed).split(xx,y,groups):
                    m=Pipeline([('pre',pre(cats,nums)),('model',Ridge(alpha=10.))]);m.fit(xx.iloc[tr],y[tr]);pred[te]=m.predict(xx.iloc[te]);assert not set(groups[tr])&set(groups[te])
                splits.append({'experiment':e,'feature':f,'split_seed':seed,**metric(y,pred)})
            print('Completed IJIE extension',e,f,flush=True)
    audit.to_csv(out/'data_audit.csv',index=False)
    for name,rows in [('participant_mean_metrics',metrics),('calibration',cal),('calibration_bands',bands),('ridge_group_permutation',perm),('ridge_split_sensitivity',splits),('ridge_feature_sets_with_intervals',abl)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    pd.DataFrame(plot).to_csv(private/'participant_mean_plot_data.csv',index=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();main(a.data_root,a.out)
