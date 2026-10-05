"""Draw every revised manuscript figure from regenerated numerical outputs.

PNG at 600 dpi and vector PDF are emitted together. Individual plotting data
are local calculation intermediates under _private and not public supplements.
"""
from pathlib import Path
import argparse
import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
COL=['#0072B2','#D55E00','#009E73','#6A3D9A','#4D4D4D'];MARK=['o','s','^','D','v']
FIELDS=['all','left','right'];PROF=[1,2,4,8]
def read(p,name):return pd.read_csv(p/(name+'.csv'))
def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    for fmt in ['png','pdf']:
        buf=io.BytesIO()
        fig.savefig(buf,format=fmt,dpi=600,bbox_inches='tight',pad_inches=.12)
        blob=buf.getvalue()
        if fmt=='png':
            from PIL import Image
            Image.open(io.BytesIO(blob)).verify()
        else:
            assert blob.rstrip().endswith(b'%%EOF')
        path=out/(name+'.'+fmt)
        temp=path.with_suffix(path.suffix+'.tmp')
        with temp.open('wb') as handle:
            handle.write(blob);handle.flush();os.fsync(handle.fileno())
        assert temp.read_bytes()==blob
        temp.replace(path)
        assert path.read_bytes()==blob
    plt.close(fig)
def error(ax,x,r,color,marker,label=None,offset=0):
    y=r['mean'].to_numpy();ax.errorbar(np.asarray(x)+offset,y,yerr=np.vstack([y-r.ci_low,r.ci_high-y]),color=color,marker=marker,linestyle='-',linewidth=1.15,markersize=4,capsize=2,label=label)
def zero(ax):ax.axhline(0,color='.4',linewidth=.8,linestyle='--',zorder=0)

def displays(p,out):
    d=read(p,'exp1_by_block_field');fig,axs=plt.subplots(1,2,figsize=(7.1,3.1),sharey=True,layout='constrained')
    for j,f in enumerate(['blur','contrast']):
        ax=axs[j]
        for k,field in enumerate(FIELDS):
            z=d[(d.feature==f)&(d.field==field)].sort_values('block');error(ax,np.arange(5),z,COL[k],MARK[k],['Full lag 1','Left lag 2','Right lag 3'][k],(k-1)*.08)
        zero(ax);ax.set_title(f'{chr(65+j)}  {f.capitalize()}');ax.set_xticks(range(5),['6','8','10','12','14'] if f=='blur' else ['−70','−30','+10','+50','+90']);ax.set_xlabel('Nominal label in fixed block order')
    axs[0].set_ylabel('Local target recovery difference');axs[1].legend(frameon=False,loc='upper right');save(fig,out,'Displays_Figure_1')
    d=read(p,'exp1_baseline_series');fig,axs=plt.subplots(1,2,figsize=(7.1,3.1),sharey=True,layout='constrained')
    for j,f in enumerate(['blur','contrast']):
        for k,s in enumerate(['baseline','manipulated']):
            z=d[(d.feature==f)&(d.series==s)].sort_values('block');error(axs[j],np.arange(5),z,COL[k],MARK[k],['Cluster start normal','Modified full field'][k])
        axs[j].set_title(f'{chr(65+j)}  {f.capitalize()}');axs[j].set_xticks(range(5),['6','8','10','12','14'] if f=='blur' else ['−70','−30','+10','+50','+90']);axs[j].set_xlabel('Nominal label in fixed block order')
    axs[0].set_ylabel('Target recovery proportion');axs[1].legend(frameon=False,loc='lower right');save(fig,out,'Displays_Figure_2')
    d=read(p,'exp1_local_pooled');fig,axs=plt.subplots(1,2,figsize=(7.1,3.1),sharey=True,layout='constrained')
    for j,f in enumerate(['blur','contrast']):
        for k,s in enumerate(['local','pooled']):
            z=d[(d.feature==f)&(d.baseline==s)].sort_values('block');error(axs[j],np.arange(5),z,COL[k],MARK[k],['Local normal reference','Pooled normal reference'][k],(k-.5)*.06)
        zero(axs[j]);axs[j].set_title(f'{chr(65+j)}  {f.capitalize()}');axs[j].set_xticks(range(5),['6','8','10','12','14'] if f=='blur' else ['−70','−30','+10','+50','+90']);axs[j].set_xlabel('Nominal label in fixed block order')
    axs[0].set_ylabel('Full field target recovery difference');axs[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2);save(fig,out,'Displays_Figure_3')
    d=read(p,'exp2_profile_means');fig,axs=plt.subplots(1,2,figsize=(7.1,3.3),sharey=True,layout='constrained')
    labs=['Homogeneous\ntrials 0–11','Two unique\ntrials 12–23','Four unique\ntrials 24–35','Eight unique\ntrials 36–47']
    for j,f in enumerate(['blur','contrast']):
        z=d[d.feature==f].sort_values('profile');error(axs[j],range(4),z,COL[j],MARK[j]);axs[j].set_title(f'{chr(65+j)}  {f.capitalize()}');axs[j].set_xticks(range(4),labs,fontsize=8);axs[j].set_xlabel('Profile and its fixed trial position')
    axs[0].set_ylabel('Target recovery proportion');save(fig,out,'Displays_Figure_4')

def architecture(out):
    fig,ax=plt.subplots(figsize=(7.1,4.6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    def box(x,y,w,h,text,fill='#F0F5F8'):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.012,rounding_size=0.012',linewidth=1,edgecolor='#34566A',facecolor=fill));ax.text(x,y,text,ha='center',va='center',fontsize=9)
    def arrow(x,y,xx,yy):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=11,color='#34566A',linewidth=1.1))
    box(.5,.92,.82,.09,'Source records with 64 recorded identifiers per dataset')
    box(.24,.72,.39,.12,'Training fold\n51 or 52 identifiers');box(.76,.72,.39,.12,'Held out fold\n12 or 13 identifiers',fill='#FFF2E8')
    arrow(.35,.865,.24,.79);arrow(.65,.865,.76,.79)
    box(.24,.49,.39,.14,'Learn imputation and encoding\nFit scaling and model\nFixed model settings')
    box(.76,.49,.39,.14,'Apply training transformations\nPredict every held out trial')
    arrow(.24,.65,.24,.57);arrow(.76,.65,.76,.57);arrow(.448,.49,.552,.49)
    box(.5,.25,.82,.13,'Repeat across five disjoint test folds\nPool out of fold predictions for MAE, RMSE and R²')
    arrow(.76,.41,.65,.325);arrow(.24,.41,.35,.325)
    ax.text(.5,.065,'All observations from an identifier stay together.\nNo test fold is used to fit preprocessing or select hyperparameters.',ha='center',va='center',fontsize=9)
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98);save(fig,out,'FIG1_validation_architecture')

def ijie(p,out):
    architecture(out);ds=[(1,'blur'),(1,'contrast'),(2,'blur'),(2,'contrast')];labels=['Exp1 blur','Exp1 contrast','Exp2 blur','Exp2 contrast']
    d=read(p,'legacy_validation_comparison');fig,ax=plt.subplots(figsize=(6.8,3.2),layout='constrained')
    for k,(kind,label) in enumerate([('rowwise','Row wise'),('participant_grouped','Identifier grouped')]):
        vals=[d[(d.experiment==e)&(d.feature==f)&(d.validation==kind)].R2.iloc[0] for e,f in ds]
        ax.bar(np.arange(4)+(k-.5)*.32,vals,width=.3,color=COL[k],label=label,hatch='' if k==0 else '//',edgecolor='white')
    zero(ax);ax.set_xticks(range(4),labels);ax.set_ylabel('Out of fold R² on legacy aggregate target');ax.legend(frameon=False,ncol=2);save(fig,out,'FIG2_legacy_validation')
    d=read(p,'primary_model_performance');models=['MeanBaseline','Ridge','RandomForest','GradientBoosting','SVR'];labs=['Mean','Ridge','RF','Boosting','SVR']
    fig,axs=plt.subplots(2,2,figsize=(7.1,5.2),sharex=True,sharey=True,layout='constrained')
    for j,((e,f),ax) in enumerate(zip(ds,axs.flat)):
        z=d[(d.experiment==e)&(d.feature==f)].set_index('model').loc[models]
        for k,m in enumerate(models):
            row=z.loc[m];ax.errorbar(k,row.R2,yerr=[[row.R2-row.R2_CI_low],[row.R2_CI_high-row.R2]],fmt=MARK[k],color=COL[k],capsize=3,markersize=5)
        zero(ax);ax.set_title(f'{chr(65+j)}  {labels[j]}');ax.set_xticks(range(5),labs);ax.set_ylabel('Trial level R²')
    save(fig,out,'FIG3_model_r2')
    d=read(p/'_private','participant_mean_plot_data');z=d[(d.experiment==2)&(d.feature=='blur')];r=read(p,'participant_mean_metrics');r2=r[(r.experiment==2)&(r.feature=='blur')].R2.iloc[0]
    fig,ax=plt.subplots(figsize=(4.9,4.1),layout='constrained');ax.scatter(z.observed,z.predicted,s=20,facecolors='none',edgecolors=COL[0],linewidths=.9);ax.plot([1.1,4.05],[1.1,4.05],color='.4',linestyle='--',linewidth=1)
    ax.set(xlabel='Observed participant mean correct targets',ylabel='Predicted participant mean correct targets',xlim=(1.1,4.05),ylim=(1.1,4.05));ax.set_aspect('equal');ax.text(.03,.96,f'Exp2 blur · 64 identifiers\nParticipant mean R² = {r2:.3f}',transform=ax.transAxes,va='top');save(fig,out,'FIG4_exp2_blur_observed_predicted')
    d=read(p,'calibration_bands');c=read(p,'calibration');fig,axs=plt.subplots(2,2,figsize=(7.1,5.2),layout='constrained')
    for j,((e,f),ax) in enumerate(zip(ds,axs.flat)):
        z=d[(d.experiment==e)&(d.feature==f)];row=c[(c.experiment==e)&(c.feature==f)].iloc[0]
        ax.plot(z.predicted,z.fit,color=COL[0],label='Calibration fit');ax.fill_between(z.predicted,z.ci_low,z.ci_high,color=COL[0],alpha=.15,label='95% fixed OOF bootstrap interval');ax.plot(z.predicted,z.predicted,color='.4',linestyle='--',linewidth=1,label='Identity')
        ax.set_title(f'{chr(65+j)}  {labels[j]}');ax.set_xlabel('Predicted trial score');ax.set_ylabel('Observed score calibration');ax.text(.03,.97,f'Intercept {row.intercept:.2f}\nSlope {row.slope:.2f}',transform=ax.transAxes,va='top',fontsize=8)
    axs[1,1].legend(frameon=False,fontsize=6.8,loc='lower right');save(fig,out,'FIG5_calibration')
    d=read(p,'ridge_feature_sets_with_intervals');sets=['conditions_only','demographics_only','conditions_demographics','conditions_technical','full','full_minus_age','full_minus_heterogeneity','full_minus_degradation'];slabs=['Conditions','Demographics','Conditions + demographics','Conditions + frame rate','Full','Full − age','Full − profile','Full − level or schedule']
    fig,axs=plt.subplots(2,2,figsize=(7.8,6.8),sharex=True,layout='constrained')
    for j,((e,f),ax) in enumerate(zip(ds,axs.flat)):
        z=d[(d.experiment==e)&(d.feature==f)].set_index('feature_set').loc[sets];vals=z.R2.to_numpy();ax.errorbar(vals,np.arange(8),xerr=np.vstack([vals-z.R2_CI_low,z.R2_CI_high-vals]),fmt='o',color=COL[0],capsize=2,markersize=4);ax.axvline(0,color='.4',ls='--',lw=.8);ax.set_yticks(range(8),slabs,fontsize=9);ax.invert_yaxis();ax.set_title(f'{chr(65+j)}  {labels[j]}');ax.set_xlabel('Trial R² (fixed OOF 95% CI)')
    save(fig,out,'FIG6_ablation')
    d=read(p,'ridge_permutation_importance');features=['condition','field','heterogeneity','gender','age','framerate'];fl=['Level or schedule','Field or position','Profile or block','Recorded gender','Age','Frame rate'];fig,axs=plt.subplots(2,2,figsize=(7.3,5.4),sharex=True,layout='constrained')
    for j,((e,f),ax) in enumerate(zip(ds,axs.flat)):
        z=d[(d.experiment==e)&(d.feature_family==f)].groupby('feature').delta_MAE.mean().reindex(features);ax.barh(range(6),z,color=COL[j],height=.6);ax.set_yticks(range(6),fl,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='.4',lw=.8);ax.set_title(f'{chr(65+j)}  {labels[j]}');ax.set_xlabel('MAE increase after permutation')
    save(fig,out,'FIG7_permutation_importance')

def age(p,out):
    d=read(p/'_private','age_plot_data')
    names=[('blur','overall','AGE_FIG1_blur_age_mean'),('contrast','overall','AGE_FIG2_contrast_age_mean'),('blur','delta','AGE_FIG3_blur_local_difference'),('contrast','delta','AGE_FIG4_contrast_local_difference')]
    for f,kind,name in names:
        z=d[(d.experiment==1)&(d.feature==f)&(d.outcome==kind)];fig,ax=plt.subplots(figsize=(6.1,3.7),layout='constrained');ax.scatter(z.age,z.value,s=24,facecolors='none',edgecolors=COL[0 if f=='blur' else 1],linewidths=.9)
        coef=np.polyfit(z.age,z.value,1);xx=np.linspace(z.age.min(),z.age.max(),100);ax.plot(xx,np.polyval(coef,xx),color=COL[0 if f=='blur' else 1],linewidth=1.5)
        if kind=='delta':zero(ax)
        ax.set_xlabel('Recorded age in years');ax.set_ylabel('Mean correct targets' if kind=='overall' else 'Manipulated minus cluster start normal\nmean correct targets');ax.set_title(f'Experiment 1 {f}')
        save(fig,out,name)
    fig,axs=plt.subplots(2,2,figsize=(7.0,4.8),sharex=True,sharey=True,layout='constrained')
    for j,((e,f),ax) in enumerate(zip([(1,'blur'),(1,'contrast'),(2,'blur'),(2,'contrast')],axs.flat)):
        z=d[(d.experiment==e)&(d.feature==f)&(d.outcome=='overall')];ax.hist(z.age,bins=[18,25,30,35,40,45,50,55,60,65],color=COL[j],edgecolor='white');ax.axvline(60,color='.3',linestyle='--',linewidth=1);ax.set_title(f'{chr(65+j)}  Exp{e} {f}');ax.set_xlabel('Recorded age in years');ax.set_ylabel('Identifier count')
    save(fig,out,'AGE_FIGS1_age_distribution')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=['Displays','IJIE','Age','all'],default='all');ap.add_argument('--results-root',required=True);ap.add_argument('--figures-root',required=True);a=ap.parse_args()
    funcs={'Displays':displays,'IJIE':ijie,'Age':age}
    for s,fn in funcs.items():
        if a.study in ['all',s]:fn(Path(a.results_root)/s,Path(a.figures_root)/s)
