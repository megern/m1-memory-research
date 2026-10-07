"""Render retained schedule results after every timed trial has ended."""
import argparse,json
from pathlib import Path


def plot(directory,output):
    summary=json.loads((directory/'summary.json').read_text())
    if not summary['completed_series']:raise ValueError('Do not render during timed trials')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    trials=summary['trials'];x=list(range(len(trials)))
    labels=[f"{t['variant']}\nround {t['round']+1}" for t in trials]
    variants=json.loads((directory/'protocol.json').read_text())['variants']
    palette=['#4466a0','#cf7932','#32816a','#855da5']
    colors={name:palette[i%len(palette)] for i,name in enumerate(variants)}
    first=next((t for t in trials if t.get('cases')),None)
    lengths=[c['prompt_tokens'] for c in first['cases']] if first else []
    prompt_label=', '.join(str(n) for n in lengths)+' token prompts' if lengths else 'Prompt lengths not recorded'
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    metrics=[('forward_seconds',1,'Forward time (seconds)',False),
             ('peak_mlx_bytes',1024**3,'MLX peak (GiB; stopped: recorded steps only)',False),
             ('layer_weight_loads',1,'Original layer weight loads (logical count)',True),
             ('max_setup_and_worker_swap_growth_bytes',1024**2,'Observed whole-system swap growth (MiB)',False)]
    for ax,(key,divisor,title,integer) in zip(axes.flat,metrics):
        for i,t in enumerate(trials):
            value=t.get(key)
            if value is None:
                ax.text(i,.03,'not recorded',rotation=90,ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
                continue
            color=colors[t['variant']] if t['completed'] else '#b74343'
            v=value/divisor
            ax.bar(i,v,color=color,hatch=None if t['completed'] else '//',alpha=.9)
            label=f'{v:.0f}' if integer else f'{v:.2f}'
            if not t['completed'] and key in {'forward_seconds','peak_mlx_bytes'}:
                label += ' (partial)'
            ax.annotate(label,(i,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(x,labels,fontsize=8)
        ax.set_title(title,fontsize=11)
        ax.set_ylim(bottom=0)
        ax.margins(y=.2)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Original BF16 Qwen3-14B on 16 GiB Apple M1: prompt scheduling\n'+prompt_label+'; two exploratory rounds',fontsize=14)
    fig.legend(handles=[Patch(color=colors[k],label=k) for k in colors]+[Patch(facecolor='#b74343',hatch='//',label='stopped attempt')],loc='lower center',bbox_to_anchor=(.5,.045),ncol=4)
    fig.text(.5,.016,'Forward excludes verification; failed partial times are not full-workload speed. MLX is not total RAM. Swap includes other apps. OS caches uncontrolled.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.095,1,.93))
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=160)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)
    svg=output.with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.directory,a.output)
