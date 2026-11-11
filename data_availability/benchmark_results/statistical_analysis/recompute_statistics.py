"""Recompute fixed-score, case-paired statistics and figures with Python only."""
from pathlib import Path
import csv
import hashlib
import json
import platform
from statistics import NormalDist
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
INPUTS = {
    'idea': BASE/'idea_evaluation/judge_outputs/per_idea_judge_aggregates.json',
    'protocol': BASE/'experiment_evaluation/judge_outputs/protocol_judge_scores.json',
}
SYSTEMS = ['SupraMAS', 'GPT-Researcher', 'GPT-5', 'Gemini-2.5-Pro-Thinking', 'Grok-4', 'DeepSeek-V3']
LABELS = ['SupraMAS', 'GPT-Researcher', 'GPT-5', 'Gemini-2.5-Pro\nThinking', 'Grok-4', 'DeepSeek-V3']
METRICS = ['strategy_overlap', 'strategy_combination', 'experimental_reproducibility', 'experimental_feasibility']
TITLES = ['Strategy overlap', 'Strategy combination', 'Protocol reproducibility', 'Protocol feasibility']
JUDGES = ['Claude', 'GPT', 'Two-judge mean']
COLORS = {'Claude': '#0072B2', 'GPT': '#A15B87'}
PALETTE = ['#252525', '#71A6BD', '#D5A368', '#A897BE', '#7EAAA0', '#BE8894']
SEED, B, P = 20261024, 20000, 100000
normal = NormalDist()
hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in INPUTS.items()}
records, seen = [], set()

def add(metric, case, system, judge, score, source, fallback=''):
    key = (metric, str(case), system, judge)
    assert key not in seen, ('Duplicate score', key)
    seen.add(key)
    score = float(score)
    assert np.isfinite(score) and 0 <= score <= 1, key
    records.append(dict(metric=metric, case=str(case), system=system, judge=judge,
                        score=score, source=source, legacy_fallback_groups=fallback))

for r in json.loads(INPUTS['idea'].read_text()):
    judge = {'J1': 'Claude', 'J2': 'GPT'}[r['judge_id']]
    for metric in METRICS[:2]:
        add(metric, r['topic'], r['system'], judge, r[metric]['score'],
            'Frozen idea aggregate', r.get('legacy_fallback_groups', 0))
for r in json.loads(INPUTS['protocol'].read_text()):
    add(r['metric'], f"L{int(r['case_id']):02d}", r['system'], r['model'],
        r['output']['res']['score'], r.get('source', ''))
assert len(records) == 1512

def dump_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def bca(x, boot):
    theta = float(x.mean())
    jack = (x.sum()-x)/(len(x)-1)
    d = jack.mean()-jack
    denom = 6*float(np.sum(d**2))**1.5
    a = float(np.sum(d**3))/denom if denom else 0.
    fraction = (np.mean(boot < theta)+np.mean(boot <= theta))/2
    z0 = normal.inv_cdf(float(np.clip(fraction, .5/B, 1-.5/B)))
    probs = []
    for alpha in [.025, .975]:
        z = z0+normal.inv_cdf(alpha)
        probs.append(normal.cdf(z0+z/(1-a*z)))
    lo, hi = np.quantile(boot, probs)
    return dict(mean=theta, ci_low=float(lo), ci_high=float(hi),
                sd_across_cases=float(x.std(ddof=1)), bootstrap_se=float(boot.std(ddof=1)), n=len(x))

def holm(rows):
    order = np.argsort([r['p_raw'] for r in rows], kind='stable')
    running = 0.
    for rank, idx in enumerate(order):
        running = max(running, min(1., (len(rows)-rank)*rows[idx]['p_raw']))
        rows[idx]['p_holm'] = running
        rows[idx]['holm_family_size'] = len(rows)

arrays, estimates, comparisons, archive = {}, [], [], {}
for mi, metric in enumerate(METRICS):
    rows = [r for r in records if r['metric'] == metric]
    cases = sorted({r['case'] for r in rows})
    assert len(cases) == (32 if mi < 2 else 31)
    lookup = {(r['case'],r['system'],r['judge']):r['score'] for r in rows}
    raw = np.array([[[lookup[c,s,j] for s in SYSTEMS] for j in JUDGES[:2]] for c in cases])
    data = np.concatenate([raw, raw.mean(axis=1, keepdims=True)], axis=1)
    arrays[metric] = data
    rng = np.random.default_rng(SEED+mi)
    indices = rng.integers(0, len(cases), (B,len(cases)))
    boot = data[indices].mean(axis=1)
    diff = data[:,:,0,None]-data[:,:,1:]
    bootdiff = boot[:,:,0,None]-boot[:,:,1:]
    archive[metric+'_indices'] = indices
    archive[metric+'_means'] = boot
    archive[metric+'_differences'] = bootdiff
    for ji, judge in enumerate(JUDGES):
        for si, system in enumerate(SYSTEMS):
            estimates.append(dict(metric=metric, judge=judge, system=system,
                                  **bca(data[:,ji,si], boot[:,ji,si]),
                                  interval='Pointwise 95% BCa', conditional_selection=False))
    # The same case signs are shared across all comparisons within a metric.
    observed = np.abs(diff.mean(axis=0))
    counts = np.zeros((3,5), dtype=int)
    prng = np.random.default_rng(SEED+100+mi)
    for start in range(0,P,5000):
        signs = prng.integers(0,2,(min(5000,P-start),len(cases)),dtype=np.int8)*2-1
        perm = (signs @ diff.reshape(len(cases),-1)/len(cases)).reshape(-1,3,5)
        counts += (np.abs(perm) >= observed-1e-12).sum(axis=0)
    for ji, judge in enumerate(JUDGES):
        for si, baseline in enumerate(SYSTEMS[1:]):
            d = diff[:,ji,si]
            stat = bca(d, bootdiff[:,ji,si])
            comparisons.append(dict(metric=metric, judge=judge, baseline=baseline,
                                    mean_difference=stat['mean'], ci_low=stat['ci_low'], ci_high=stat['ci_high'],
                                    n=len(cases), wins=int((d>1e-10).sum()), ties=int((np.abs(d)<=1e-10).sum()),
                                    losses=int((d < -1e-10).sum()), p_raw=float((counts[ji,si]+1)/(P+1)),
                                    conditional_selection=False))
holm([r for r in comparisons if r['judge'] != JUDGES[2]])
holm([r for r in comparisons if r['judge'] == JUDGES[2]])
for r in comparisons:
    p = r['p_holm']
    r['significance'] = ('***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns')

dump_csv('SourceData_case_judge_scores.csv', records)
dump_csv('mean_scores_and_95BCa_intervals.csv', estimates)
dump_csv('paired_differences_and_significance.csv', comparisons)
np.savez_compressed(OUT/'bootstrap_samples.npz', **archive)
(OUT/'statistical_results.json').write_text(json.dumps(dict(estimates=estimates, comparisons=comparisons),ensure_ascii=False,indent=2))

plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':6.5, 'axes.labelsize':7, 'axes.titlesize':7, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.linewidth':.6, 'legend.frameon':False,
    'svg.fonttype':'none', 'pdf.fonttype':42})

def save(fig, name):
    for ext in ['png','pdf','svg','tiff']:
        kw = {'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
        fig.savefig(OUT/f'{name}.{ext}',dpi=600 if ext=='tiff' else 400,facecolor='white',**kw)
    plt.close(fig)

def title(ax,mi):
    ax.set_title(f'{chr(98+mi)}  {TITLES[mi]}'+f'  (n={32 if mi<2 else 31})',loc='left',pad=8)
    ax.tick_params(width=.55,length=2)

def foot(fig, text):
    fig.text(.5,.043,text,ha='center',fontsize=5.6)

fig,axes=plt.subplots(2,2,figsize=(183/25.4,158/25.4))
fig.subplots_adjust(left=.17,right=.985,top=.90,bottom=.15,hspace=.42,wspace=.70)
for mi,(metric,ax) in enumerate(zip(METRICS,axes.flat)):
    data=arrays[metric]
    for si,s in enumerate(SYSTEMS):
        for ji,j in enumerate(JUDGES[:2]):
            y=si+(-.18 if ji==0 else .18)
            jitter=np.random.default_rng(SEED+mi*100+si).uniform(-.07,.07,len(data))
            ax.scatter(data[:,ji,si],y+jitter,s=3.8,c=COLORS[j],alpha=.25,linewidths=0,zorder=1)
            e=next(r for r in estimates if (r['metric'],r['judge'],r['system'])==(metric,j,s))
            ax.errorbar(e['mean'],y,xerr=[[e['mean']-e['ci_low']],[e['ci_high']-e['mean']]],fmt='D' if si==0 else 'o',color=COLORS[j],ms=3.6,mec='white',mew=.35,capsize=2,elinewidth=1.05,zorder=4)
        if si<5:ax.axhline(si+.5,color='#ededed',lw=.5,zorder=0)
    ax.set_yticks(range(6),LABELS);ax.set_ylim(5.55,-.55)
    ax.set_xlim(0,1);ax.set_xticks(np.arange(0,1.01,.2));ax.set_xlabel('Score');title(ax,mi)
fig.suptitle('Automatic evaluation | Claude and GPT',fontsize=10,y=.978)
handles=[Line2D([],[],color=COLORS[j],marker='o',lw=1,ms=3.5,label=j) for j in JUDGES[:2]]
handles += [Line2D([],[],color='#555555',marker=m,ls='none',ms=3.5,label=l) for m,l in [('D','SupraMAS'),('o','Baselines')]]
fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.5,.066),fontsize=7)
foot(fig,'Small points: cases. Large markers: means. Bars: 95% BCa CI (20,000 case bootstrap resamples).')
save(fig,'01_four_metrics_Claude_and_GPT')

fig,axes=plt.subplots(2,2,figsize=(183/25.4,158/25.4))
fig.subplots_adjust(left=.17,right=.97,top=.90,bottom=.15,hspace=.43,wspace=.72)
for mi,(metric,ax) in enumerate(zip(METRICS,axes.flat)):
    rows=[r for r in comparisons if r['metric']==metric and r['judge'] in JUDGES[:2]]
    lo=min(0,min(r['ci_low'] for r in rows));hi=max(r['ci_high'] for r in rows)
    span=hi-lo
    ax.set_xlim(lo-.09*span,hi+.48*span)
    ax.axvline(0,color='#777777',lw=.7,ls='--')
    for si,s in enumerate(SYSTEMS[1:]):
        for ji,j in enumerate(JUDGES[:2]):
            r=next(r for r in rows if r['judge']==j and r['baseline']==s)
            y=si+(-.17 if ji==0 else .17);mean=r['mean_difference']
            ax.errorbar(mean,y,xerr=[[mean-r['ci_low']],[r['ci_high']-mean]],fmt='o',color=COLORS[j],ms=3.4,capsize=2,lw=1)
            label=r['significance']
            ax.text(hi+.12*span,y,label,va='center',fontsize=5.5,color=COLORS[j])
        if si<4:ax.axhline(si+.5,color='#ededed',lw=.5,zorder=0)
    ax.set_yticks(range(5),LABELS[1:]);ax.set_ylim(4.55,-.55)
    ax.set_xlabel('Mean paired difference (SupraMAS \u2212 baseline)');title(ax,mi)
fig.suptitle('Paired effects | SupraMAS versus each baseline',fontsize=10,y=.978)
fig.legend(handles=handles[:2],ncol=2,loc='lower center',bbox_to_anchor=(.5,.072),fontsize=7)
foot(fig,'95% BCa CI; two-sided paired sign-flip (100,000); Holm across 40 judge-specific tests. * <0.05; ** <0.01; *** <0.001.')
save(fig,'02_paired_differences_and_significance')

fig,axes=plt.subplots(4,2,figsize=(183/25.4,205/25.4))
fig.subplots_adjust(left=.09,right=.98,top=.925,bottom=.14,hspace=.72,wspace=.30)
for mi,metric in enumerate(METRICS):
    boot=archive[metric+'_means']
    limits=(float(np.min(boot[:,:2])),float(np.max(boot[:,:2])))
    pad=(limits[1]-limits[0])*.06
    bins=np.linspace(limits[0]-pad,limits[1]+pad,115)
    for ji,j in enumerate(JUDGES[:2]):
        ax=axes[mi,ji]
        for si in range(5,-1,-1):
            vals=boot[:,ji,si]
            density,edges=np.histogram(vals,bins=bins,density=True)
            ax.stairs(density,edges,color=PALETTE[si],lw=1.15 if si==0 else .75,fill=False,zorder=10-si)
            ax.axvline(arrays[metric][:,ji,si].mean(),color=PALETTE[si],lw=.55,ls=':',alpha=.8)
        ax.set_xlim(bins[0],bins[-1]);ax.set_xlabel('Bootstrap mean score');ax.set_ylabel('Density')
        ax.set_title(f'{chr(97+2*mi+ji)}  {TITLES[mi]}'+f' | {j}',loc='left',pad=6)
        ax.tick_params(width=.55,length=2)
fig.suptitle('Case bootstrap | Sampling distributions of system means',fontsize=10,y=.978)
hh=[Line2D([],[],color=c,lw=1.3,label=l.replace('\n',' ')) for c,l in zip(PALETTE,LABELS)]
fig.legend(handles=hh,ncol=3,loc='lower center',bbox_to_anchor=(.5,.065),fontsize=6.5)
foot(fig,'20,000 paired case resamples. Dotted lines: observed means. Topics n=32; protocol cases n=31.')
save(fig,'03_bootstrap_sampling_distributions')

fig,axes=plt.subplots(2,2,figsize=(183/25.4,150/25.4))
fig.subplots_adjust(left=.17,right=.97,top=.90,bottom=.14,hspace=.43,wspace=.72)
for mi,(metric,ax) in enumerate(zip(METRICS,axes.flat)):
    rows=[r for r in comparisons if r['metric']==metric and r['judge']==JUDGES[2]]
    lo=min(0,min(r['ci_low'] for r in rows));hi=max(r['ci_high'] for r in rows);span=hi-lo
    ax.set_xlim(lo-.09*span,hi+.48*span);ax.axvline(0,color='#777777',lw=.7,ls='--')
    for si,s in enumerate(SYSTEMS[1:]):
        r=next(r for r in rows if r['baseline']==s);mean=r['mean_difference']
        ax.errorbar(mean,si,xerr=[[mean-r['ci_low']],[r['ci_high']-mean]],fmt='D',color='#404040',ms=3.6,capsize=2,lw=1)
        ax.text(hi+.12*span,si,r['significance'],va='center',fontsize=5.5)
    ax.set_yticks(range(5),LABELS[1:]);ax.set_ylim(4.55,-.55)
    ax.set_xlabel('Mean paired difference (SupraMAS \u2212 baseline)');title(ax,mi)
fig.suptitle('Two-judge mean | Paired effects',fontsize=10,y=.978)
foot(fig,'Average the fixed judges within each case first; 95% BCa CI; two-sided sign-flip; separate Holm family of 20 tests.')
save(fig,'04_two_judge_mean_paired_differences')

plan = dict(seed=SEED, bootstrap=B, permutations=P, CI='Pointwise 95% BCa, midrank bias correction',
    resampling_unit='Topic (idea n=32) or case (protocol n=31); paired indices shared across systems and judges',
    test='Two-sided paired sign-flip of mean difference; plus-one Monte Carlo p',
    multiplicity='Main two-judge-mean analysis: Holm across 20 metric x baseline comparisons; judge-specific analysis: separate Holm family of 40',
    scoring_rule='Analyze the recorded fixed scores without rescoring or selecting among repeated runs.',
    fixed_judges='Two fixed judges, not independent subject replicates; no inference to a population of judges',
    idea_input='Final per-idea aggregates from valid group-level evaluations.',
    source_sha256=hashes, versions=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
    artifact_type='Quantitative grid; fixed-score comparison and case-sampling uncertainty',
    exports='PNG 400 dpi, TIFF 600 dpi LZW, editable SVG/PDF; figures 183 mm wide',
    score_changes=False, API_calls=0)
existing_plan = json.loads((OUT/'analysis_plan_and_sources.json').read_text())
if 'scoring_revision' in existing_plan:
    plan['scoring_revision'] = existing_plan['scoring_revision']
(OUT/'analysis_plan_and_sources.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in INPUTS.items())
validation=dict(case_points=len(records), means=len(estimates), comparisons=len(comparisons),
    judge_specific_family=40, pooled_family=20, source_hashes_unchanged=True,
    complete_balanced_panels=True, no_duplicate_atomic_scores=True,
    all_intervals_finite=all(np.isfinite([r['ci_low'],r['ci_high']]).all() for r in estimates+comparisons),
    supra_gpt_reproducibility_mean=next(r['mean'] for r in estimates if (r['system'],r['judge'],r['metric'])==('SupraMAS','GPT',METRICS[2])))
assert abs(validation['supra_gpt_reproducibility_mean']-.6941935483870968)<1e-12

print(json.dumps(validation,ensure_ascii=False,indent=2))
print('POOLED COMPARISONS')
for r in comparisons:
    if r['judge']==JUDGES[2]:print(r['metric'],r['baseline'],round(r['mean_difference'],5),round(r['p_holm'],5),r['significance'])
