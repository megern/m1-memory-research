"""Render a completed consecutive-process control after timed model work ends."""
import argparse,json
from pathlib import Path

def plot(directory,output):
    summary=json.loads((directory/'summary.json').read_text())
    if not summary['completed_series']:raise ValueError('Complete pair required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trials=summary['trials'];fig,axes=plt.subplots(2,2,figsize=(11,8))
    panels=[('time_to_first_nonwhitespace_stdout_seconds',1,'First nonwhite stdout (seconds)'),('elapsed_seconds',1,'Total process time (seconds)'),('observed_process_tree_rss_peak_bytes',2**30,'Sampled process RSS peak (GiB)'),('prompt',1000,'Runtime prompt evaluation (seconds)')]
    for ax,(key,scale,title) in zip(axes.flat,panels):
        values=[(next(v['milliseconds'] for v in t['parsed_timing_lines'] if v['kind']=='prompt eval time') if key=='prompt' else t[key])/scale for t in trials]
        ax.bar([0,1],values,color=['#4466a0','#32816a'])
        ax.set_xticks([0,1],['First fresh process','Second fresh process'])
        ax.set_title(title);ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for index,value in enumerate(values):ax.annotate(f'{value:.2f}',(index,value),xytext=(0,4),textcoords='offset points',ha='center')
    fig.suptitle('Llama 3.3 70B IQ2_XXS on Apple M1 / 16 GiB: consecutive launches\nOne observed pair; quantized checkpoint, CPU mmap',fontsize=13)
    fig.text(.5,.035,'One checksum before first process; no rehash or application KV/session cache between them. OS caches and other apps uncontrolled.',ha='center',fontsize=8)
    fig.text(.5,.014,'First stdout includes loading/prefill. This is not a cold/warm proof, stable speedup or measurement of surviving cache pages.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.065,1,.93));output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['png','svg']:fig.savefig(output.with_suffix('.'+extension),dpi=160)
    plt.close(fig);svg=output.with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.directory,a.output)
