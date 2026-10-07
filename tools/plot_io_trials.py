"""Plot all attempted jobs; stopped durations are never presented as speedups."""
import argparse
import json
from pathlib import Path


def plot(summary_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    summary=json.loads(summary_path.read_text())
    trials=summary['trials']
    labels=[f'R{t["round"]+1} {t["mode"]}' for t in trials]
    y=np.arange(len(trials))
    colors=['#247b83' if t['completed'] else '#bc4749' for t in trials]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    axes[0].barh(y,[t.get('worker_seconds',0) for t in trials],color=colors)
    axes[0].set_xlabel('Worker elapsed seconds (verification excluded)')
    axes[0].set_title('Every attempted job; stopped time is incomplete')
    axes[0].set_yticks(y,labels)
    axes[0].invert_yaxis()
    axes[0].legend(handles=[Patch(color='#247b83',label='Completed'),Patch(color='#bc4749',label='Stopped/failed')],loc='upper right')
    verification=[t.get('verification_swap_delta_bytes',0)/1024**2 for t in trials]
    growth=[t.get('max_system_swap_growth_bytes',0)/1024**2 for t in trials]
    axes[1].barh(y-.17,verification,height=.32,color='#a5b5bd',label='Verification net delta')
    axes[1].barh(y+.17,growth,height=.32,color=colors,label='Worker maximum growth')
    axes[1].axvline(512,color='#333333',linestyle='--',linewidth=1,label='Worker stop threshold')
    axes[1].set_xlabel('Whole-system swap MiB; includes other apps')
    axes[1].set_title('Verification and worker use separate baselines')
    axes[1].set_yticks(y,labels)
    axes[1].invert_yaxis()
    axes[1].legend(handles=[Line2D([0],[0],color='#333333',linestyle='--',label='Worker stop threshold'),
        Patch(color='#a5b5bd',label='Verification net delta'),
        Patch(color='#247b83',label='Worker max: completed'),
        Patch(color='#bc4749',label='Worker max: stopped')],loc='lower right',fontsize=8)
    fig.suptitle('Original BF16 Qwen3-14B on M1 16 GiB: exploratory I/O controls',fontsize=14)
    fig.text(.5,.015,'OS caches uncontrolled; checksum reads warm caches. Two rounds; no general speed or stability claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94))
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=180)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.summary,a.output)
