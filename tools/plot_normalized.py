"""Visualize all projected-policy trials without attributing arithmetic to the LLM."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'runs/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(summary, output):
    output.mkdir(parents=True,exist_ok=True)
    entries=summary['entries']
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,metric,title in zip(axes,['strict','outer_fence_allowed'],['Strict JSON classification','Classification allowing outer JSON fence']):
        values=[e[metric]['accuracy']*100 for e in entries]
        bars=ax.bar(range(4),values,color=['#7d8b99']+['#156c89']*3,width=.65)
        ax.axhline(summary['direct_rule_baseline']['accuracy']*100,color='#ac6236',ls='--',lw=1)
        ax.set_xticks(range(4),['Base','Seed 17','Seed 23','Seed 41'])
        ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100])
        ax.set_title(title,fontsize=10);ax.set_ylabel('Accuracy (%)')
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+2,f'{value:.1f}',ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Code projection + Qwen3 0.6B LoRA: all seeds, 96 synthetic cases',fontsize=12)
    fig.text(.5,.055,'Dashed line: direct rule baseline. Python performs numeric comparisons. Only eight possible Boolean states.',ha='center',fontsize=8)
    fig.text(.5,.02,'This exploratory dataset changes sampling and wording too; it does not measure general incident or arithmetic reasoning.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.09,1,.93])
    fig.savefig(output/'normalized-accuracy.png',dpi=180)
    fig.savefig(output/'normalized-accuracy.svg')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(json.loads(a.summary.read_text()),a.output)
