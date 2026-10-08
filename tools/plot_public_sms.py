"""Standalone paired-quality scientific figure; no timings or generalization claim."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paired=json.loads((a.results/'paired-training-control-analysis.json').read_text());quality=json.loads((a.results/'quality-analysis.json').read_text())
    seeds=['17','23','41'];x=np.arange(3);old=[paired['per_seed'][s]['old_macro_f1'] for s in seeds];new=[paired['per_seed'][s]['new_macro_f1'] for s in seeds]
    fig,(left,right)=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
    left.bar(x-.18,old,.36,label='120 microbatches / no accumulation',color='#718096');left.bar(x+.18,new,.36,label='480 microbatches / accumulation 4',color='#167e86')
    left.axhline(quality['metrics']['lexical']['macro_f1'],color='#b56f14',linestyle='--',label='Naive Bayes');left.axhline(quality['metrics']['base']['macro_f1'],color='#555555',linestyle=':',label='Original base')
    left.set(xticks=x,xticklabels=[f'Seed {s}' for s in seeds],ylim=(0,1.08),ylabel='Macro-F1',title='Same fresh test, all three seeds');left.legend(fontsize=8,loc='lower left')
    labels=seeds+['Mean'];points=[paired['per_seed'][s]['paired_delta'] for s in seeds]+[paired['new_seed_mean_macro_f1']-paired['old_seed_mean_macro_f1']];bounds=[paired['per_seed'][s]['paired_case_bootstrap_ci95'] for s in seeds]+[paired['mean_paired_delta_case_ci95']]
    right.errorbar(points,np.arange(4),xerr=np.array([[v-lo for v,(lo,hi) in zip(points,bounds)],[hi-v for v,(lo,hi) in zip(points,bounds)]]),fmt='o',color='#167e86',capsize=4)
    right.axvline(0,color='#555555',linewidth=1);right.set(yticks=np.arange(4),yticklabels=['Seed '+s if s!='Mean' else s for s in labels],xlabel='New minus old Macro-F1',title='Paired case-bootstrap 95% intervals');right.invert_yaxis();right.grid(axis='x',alpha=.2)
    fig.suptitle('Local original-BF16 0.6B SMS adaptation on M1 / 16 GiB',fontsize=13)
    fig.supxlabel('256 held-out representatives, 26 spam. Exploratory two-regime comparison; fixed seeds, one historical English corpus.',fontsize=8)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'):fig.savefig(a.output.with_suffix('.'+extension),dpi=180)
    plt.close(fig)
if __name__=='__main__':main()
