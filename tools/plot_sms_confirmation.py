"""Descriptive figure of fixed-model paired policy contrasts, not seed-population CIs."""
import argparse,json
from pathlib import Path

def plot(pilot,confirmation,output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 old=json.loads(pilot.read_text());new=json.loads(confirmation.read_text())
 if new['completed_cells']!=16:raise ValueError('Complete confirmation required')
 factors=['iterations','learning_rate','gradient_accumulation_steps']
 labels=['120 → 480 microbatches','Learning rate 0.0001 → 0.0002','Accumulation 1 → 4']
 fig,ax=plt.subplots(figsize=(11,6));colors=['#64748b','#2563eb','#0f766e']
 for i,seed in enumerate((17,23,41)):
  rows=[old['contrasts'][factor] if seed==17 else new['contrasts'][f'seed{seed}:{factor}'] for factor in factors]
  points=[r['point']*100 for r in rows];intervals=[r['paired_case_ci95'] if seed==17 else r['conditional_paired_case_ci95'] for r in rows]
  errors=[[points[j]-intervals[j][0]*100 for j in range(3)],[intervals[j][1]*100-points[j] for j in range(3)]]
  ax.errorbar([j+(i-1)*.15 for j in range(3)],points,yerr=errors,fmt='o',capsize=4,color=colors[i],label=f'Seed {seed}'+(' (pilot)' if seed==17 else ' (confirmation)'))
 ax.axhline(0,color='#111827',lw=1);ax.set_xticks(range(3),labels);ax.set_ylabel('Macro-F1 difference (percentage points)');ax.grid(axis='y',alpha=.2);ax.legend(loc='best')
 ax.set_title('Policy effects on reused validation data',fontsize=16,pad=18)
 fig.text(.5,.025,'128 messages, 13 spam. 95% shared paired-case bootstrap intervals conditional on fixed models.\nIntervals exclude population seed uncertainty; accumulation also changes optimizer-update count.',ha='center',fontsize=10)
 fig.tight_layout(rect=(0,.08,1,1));output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output.with_suffix('.png'),dpi=160);fig.savefig(output.with_suffix('.svg'));plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('pilot','confirmation','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();plot(a.pilot,a.confirmation,a.output)
