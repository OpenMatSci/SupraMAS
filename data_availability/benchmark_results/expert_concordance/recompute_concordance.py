"""Fixed-system concordance and conditional case-sampling uncertainty."""
from pathlib import Path
from statistics import NormalDist
from itertools import permutations
import csv, hashlib, json, platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
MODELS=['SupraMAS','GPT-Researcher','GPT-5','Gemini-2.5-Pro-Thinking','Grok-4','DeepSeek-V3']
KEYS=['supramas','gpt_researcher','gpt-5','gemini-2.5-pro-thinking','grok-4','deepseek-v3']
SHORT=['SupraMAS','GPT-Researcher','GPT-5','Gemini-2.5-Pro','Grok-4','DeepSeek-V3']
METRICS=['strategy_overlap','strategy_combination','experimental_reproducibility','experimental_feasibility']
DIMS=['Content Completeness','Theme Relevance','Technical Innovation','Technical Feasibility','Application Value']
B=20000
SEED=20261025
rng=np.random.default_rng(SEED)
normal=NormalDist()

def ranks(x):
    x=np.asarray(x)
    return 1+(x[...,None,:]<x[...,:,None]).sum(-1)+.5*((x[...,None,:]==x[...,:,None]).sum(-1)-1)

def rho(x,y):
    a=ranks(x);b=ranks(y)
    a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True)
    return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum(-1))

def tau(x,y):
    i,j=np.triu_indices(len(x),1)
    a=np.sign(x[i]-x[j]);b=np.sign(y[i]-y[j])
    return float((a*b).sum()/np.sqrt((a!=0).sum()*(b!=0).sum()))

def summary(x,boot,method='BCa'):
    theta=float(np.mean(x))
    if method=='BCa':
        jack=(x.sum()-x)/(len(x)-1);d=jack.mean()-jack
        den=6*np.sum(d*d)**1.5
        acc=float(np.sum(d**3)/den) if den else 0.
        frac=(np.mean(boot<theta)+np.mean(boot<=theta))/2
        z0=normal.inv_cdf(float(np.clip(frac,.5/B,1-.5/B)))
        probs=[]
        for q in [.025,.975]:
            z=z0+normal.inv_cdf(q);probs.append(normal.cdf(z0+z/(1-acc*z)))
    else:probs=[.025,.975]
    lo,hi=np.quantile(boot,probs)
    return dict(mean=theta,ci_low=float(lo),ci_high=float(hi),bootstrap_se=float(boot.std(ddof=1)),n=len(x),ci_method=method)

def dump(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

idea_path=ROOT/'benchmark_results/idea_evaluation/judge_outputs/per_idea_judge_aggregates.json'
protocol_path=ROOT/'benchmark_results/experiment_evaluation/judge_outputs/protocol_judge_scores.json'
idea=json.loads(idea_path.read_text());protocol=json.loads(protocol_path.read_text())
expert=json.loads((OUT/'expert_records.json').read_text())
topics=sorted({r['topic'] for r in idea});cases=sorted({int(r['case_id']) for r in protocol})
experts=sorted({r['expert'] for r in expert})
il={(r['topic'],r['system'],r['judge_id'],m):float(r[m]['score']) for r in idea for m in METRICS[:2]}
pl={(int(r['case_id']),r['system'],r['model'],r['metric']):float(r['output']['res']['score']) for r in protocol}
el={(r['topic'],r['system'],r['expert'],r['dimension']):r['score'] for r in expert if r['system'] in KEYS}
assert len(il)==768 and len(pl)==744 and len(el)==4800
assert len(topics)==32 and len(cases)==31 and len(experts)==5
assert set(topics)=={r['topic'] for r in expert}
I=np.array([[[[il[t,m,j,k] for k in METRICS[:2]] for j in ['J1','J2']] for m in MODELS] for t in topics])
P=np.array([[[[pl[t,m,j,k] for k in METRICS[2:]] for j in ['Claude','GPT']] for m in MODELS] for t in cases])
E=np.array([[[[el[t,m,e,d] for d in DIMS] for e in experts] for m in KEYS] for t in topics])
assert np.isfinite(I).all() and np.isfinite(P).all() and np.isfinite(E).all()
assert I.min()>=0 and I.max()<=1 and P.min()>=0 and P.max()<=1
assert E.min()>=0 and E.max()<=5
# A shared topic index retains dependence between the two idea metrics and expert scores.
ti=rng.integers(0,32,(B,32));pi=rng.integers(0,31,(B,31))
ib=I[ti].mean(1);pb=P[pi].mean(1)
et=E.mean((2,3));eb=et[ti].mean(1)
am=np.concatenate([I.mean((0,2)),P.mean((0,2))],axis=-1)
ab=np.concatenate([ib.mean(2),pb.mean(2)],axis=-1)
ai=am.mean(-1);aib=ab.mean(-1);em=et.mean(0)
observed=float(rho(ai,em));kt=tau(ai,em)
bootrho=rho(aib,eb)
perms=np.array(list(permutations(range(6))))
nullrho=rho(ai[perms],em)
exact_p=float(np.mean(abs(nullrho)>=abs(observed)-1e-12))
ranking=[]
for j,m in enumerate(MODELS):
    ranking.append(dict(system=m,**{k:float(am[j,kidx]) for kidx,k in enumerate(METRICS)},automatic_index=float(ai[j]),automatic_rank=int(ranks(-ai)[j]),expert_mean=float(em[j]),expert_rank=int(ranks(-em)[j])))
dump('rank_concordance.csv',ranking)
estimates=[]
for k,m in enumerate(METRICS):
    arr=I.mean(2)[:,:,k] if k<2 else P.mean(2)[:,:,k-2]
    for j,s in enumerate(MODELS):estimates.append(dict(track='automatic',metric=m,system=s,**summary(arr[:,j],ab[:,j,k])))
for j,s in enumerate(MODELS):
    estimates.append(dict(track='expert',metric='Overall',system=s,**summary(et[:,j],eb[:,j])))
    estimates.append(dict(track='automatic',metric='Four-metric auxiliary index',system=s,mean=float(ai[j]),ci_low=float(np.quantile(aib[:,j],.025)),ci_high=float(np.quantile(aib[:,j],.975)),bootstrap_se=float(aib[:,j].std(ddof=1)),n='32 topics + 31 protocol cases',ci_method='Percentile, joint block bootstrap'))
dump('conditional_mean_intervals.csv',estimates)
# Topic-by-expert crossed resampling is a sensitivity analysis, with five experts only.
cross=np.empty((B,6))
for start in range(0,B,500):
    n=min(500,B-start);ei=rng.integers(0,5,(n,5))
    cross[start:start+n]=E[ti[start:start+n][:,:,None,None],np.arange(6)[None,None,:,None],ei[:,None,None,:],:].mean((1,3,4))
crossrho=rho(aib,cross)
crossrows=[dict(system=s,mean=float(em[j]),ci_low=float(np.quantile(cross[:,j],.025)),ci_high=float(np.quantile(cross[:,j],.975)),ci_method='Crossed topic-expert percentile sensitivity') for j,s in enumerate(MODELS)]
dump('expert_crossed_sensitivity.csv',crossrows)
sens=[]
for k,m in enumerate(METRICS):sens.append(dict(analysis='Individual automatic metric',variant=m,rho=float(rho(am[:,k],em))))
for j,s in enumerate(['Claude','GPT']):
    aj=np.concatenate([I.mean(0)[:,j,:],P.mean(0)[:,j,:]],axis=-1).mean(-1)
    sens.append(dict(analysis='Single judge index',variant=s,rho=float(rho(aj,em))))
for j,s in enumerate(MODELS):
    keep=np.arange(6)!=j;sens.append(dict(analysis='Leave one system out',variant=s,rho=float(rho(ai[keep],em[keep]))))
for k,m in enumerate(METRICS):sens.append(dict(analysis='Leave one metric out',variant=m,rho=float(rho(np.delete(am,k,axis=1).mean(1),em))))
dump('concordance_sensitivity.csv',sens)
rankboot=ranks(-aib);expert_rankboot=ranks(-eb)
stability=[dict(system=s,automatic_top1_frequency=float(np.mean(rankboot[:,j]==1)),expert_top1_frequency=float(np.mean(expert_rankboot[:,j]==1)),automatic_rank_p025=float(np.quantile(rankboot[:,j],.025)),automatic_rank_p975=float(np.quantile(rankboot[:,j],.975))) for j,s in enumerate(MODELS)]
dump('ranking_stability.csv',stability)
# All expert source values must reconcile with the currently reported topic summaries.
with (ROOT/'expert_evaluation/SourceData_expert_topic_means.csv').open(encoding='utf-8-sig') as f:
    old=list(csv.DictReader(f))
lookup={(int(r['topic_id']),r['model'],r['metric']):float(r['topic_mean_score']) for r in old}
for t_idx,t in enumerate(topics):
    topic_id=next(int(r['topic_id']) for r in expert if r['topic']==t)
    for j,m in enumerate(KEYS):
        for k,d in enumerate(DIMS):assert abs(E[t_idx,j,:,k].mean()-lookup[topic_id,m,d])<1e-12
with (ROOT/'benchmark_results/statistical_analysis/mean_scores_and_95BCa_intervals.csv').open(encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        if r['judge']=='Two-judge mean':assert abs(am[MODELS.index(r['system']),METRICS.index(r['metric'])]-float(r['mean']))<1e-12
results=dict(seed=SEED,bootstrap_resamples=B,n_systems=6,idea_topics=32,protocol_cases=31,n_experts=5,
 spearman_rho=observed,kendall_tau_b=kt,exact_permutation_p_two_sided=exact_p,exact_permutations=720,
 rho_case_bootstrap_percentile95=np.quantile(bootrho,[.025,.975]).tolist(),
 rho_crossed_expert_sensitivity_percentile95=np.quantile(crossrho,[.025,.975]).tolist(),
 case_bootstrap_rho_positive_frequency=float(np.mean(bootrho>0)),
 scoring_index='Equal mean of four metric means, with equal mean of Claude and GPT within each metric.',
 resampling='Shared 32 topic indices across idea metrics, all systems, both judges and expert track; separate shared 31 protocol-case indices across protocol metrics, systems and judges. Models are held fixed.',
 interpretation='Conditional case-resampling intervals under an exchangeable-case working model, not coverage guarantees for all materials tasks. Rank intervals are percentile stability summaries. Exact label-permutation p is supplementary under an exchangeability null for the six system labels, not population inference over models.',
 source_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in [('idea',idea_path),('protocol',protocol_path)]},
 validation={'expert_cells_match_paper':True,'automatic_means_match_current_analysis':True,'shared_idea_expert_topics_exact':True},
 software={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__})
(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
np.savez_compressed(OUT/'bootstrap_distributions.npz',automatic_index=aib,expert_mean=eb,expert_crossed=cross,rho=bootrho,rho_crossed=crossrho)

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(183/25.4,83/25.4));fig.subplots_adjust(left=.09,right=.98,bottom=.19,top=.84,wspace=.43)
ax=axs[0]
ax.plot([.6,6.4],[.6,6.4],ls='--',lw=.7,c='#BAC6D2')
for j,s in enumerate(SHORT):
    x=ranks(-ai)[j];y=ranks(-em)[j];ax.scatter(x,y,s=22,c='#285E85' if j==0 else '#8196A9')
    ax.annotate(s,(x,y),xytext=(4,5 if j!=3 else -11),textcoords='offset points',fontsize=5.6)
ax.set(xlim=(.6,7),ylim=(6.5,.5),xticks=range(1,7),yticks=range(1,7),xlabel='Automatic index rank',ylabel='Expert overall rank')
ax.set_title(f'a  System-level concordance\nSpearman rho = {observed:.3f}; n = 6',loc='left',fontsize=8)
ax=axs[1]
vals,counts=np.unique(np.round(bootrho,10),return_counts=True)
ax.vlines(vals,0,counts/B,color='#285E85',lw=2)
ax.axvline(observed,color='#243F53',ls='--',lw=.9)
ax.set(xlim=(-.05,1.05),xlabel='Bootstrap Spearman rho',ylabel='Resampling frequency')
lo,hi=results['rho_case_bootstrap_percentile95']
ax.set_title(f'b  Conditional ranking stability\n20,000 resamples; 95% range [{lo:.3f}, {hi:.3f}]',loc='left',fontsize=8)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'concordance_updated.{ext}',dpi=400,facecolor='white')
plt.close(fig)

print(json.dumps(results,ensure_ascii=False,indent=2))
