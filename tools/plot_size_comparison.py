"""Plot all raw-numeric size trials; reuse of scored data is explicitly exploratory."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'runs/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root, output):
    small=json.loads((root/'results/confirmatory-summary.json').read_text())['tasks']['auth']['entries']
    large=json.loads((root/'results/size-comparison-summary.json').read_text())['entries']
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,entries,title in zip(axes,[small,large],['Qwen3 0.6B','Qwen3 1.7B']):
        values=[e['outer_fence_allowed']['accuracy']*100 for e in entries]
        bars=ax.bar(range(4),values,color=['#7d8b99']+['#156c89']*3,width=.65)
        ax.set_xticks(range(4),['Base','Seed 17','Seed 23','Seed 41'])
        ax.set_ylim(0,100)
        ax.set_title(title)
        ax.set_ylabel('Classification accuracy (%)')
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+2,f'{value:.1f}',ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Raw-numeric authentication policy: all seeds, same 96 test cases')
    fig.text(.5,.015,'Outer JSON fence allowed. The later 1.7B series reuses an already scored test; this is an exploratory comparison.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.93])
    fig.savefig(output/'size-comparison.png',dpi=180)
    fig.savefig(output/'size-comparison.svg')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.root,a.output)
