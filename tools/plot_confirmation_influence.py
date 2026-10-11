"""Recreate the post-run diagnostic figure from published JSON; no training."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
r=json.loads((root/'results/confirmation-influence-20261011.json').read_text())
keys=['iterations','learning_rate','gradient_accumulation_steps'];labels=['Exposure: 120 → 480','Learning rate: 0.0001 → 0.0002','Accumulation: 1 → 4']
fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
for i,key in enumerate(keys):
 lo=r['leave_one_out'][key]['minimum']*100;hi=r['leave_one_out'][key]['maximum']*100;point=r['baseline_two_seed_mean_effects'][key]*100
 ax.plot([lo,hi],[i,i],linewidth=6,color='#c7d2fe',solid_capstyle='round');ax.scatter([point],[i],color='#1e3a8a',s=80,zorder=3)
 ax.text(hi+.7,i,f'{lo:+.2f} to {hi:+.2f}',va='center',fontsize=10)
ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.axvline(0,color='#94a3b8',linewidth=1);ax.set_xlim(-17,30);ax.set_xlabel('Macro-F1 effect (percentage points)');ax.set_title('Case-influence sensitivity on reused validation data',loc='left',weight='bold',pad=16)
fig.text(.015,-.06,'Two fixed seeds · 128 paired cases · leave-one-out ranges, not confidence intervals\nPost-run descriptive diagnostic; no fresh held-out evaluation. Megern',fontsize=9,color='#475569')
fig.savefig(root/'results/confirmation-influence-20261011.png',dpi=180,bbox_inches='tight');plt.close(fig)
