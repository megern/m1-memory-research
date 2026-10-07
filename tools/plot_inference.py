"""Plot measured memory and page-in counters, keeping system and process separate."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'runs/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(report, output):
    rows=report['measurements']
    if not rows:
        raise ValueError('No measurements recorded; refusing an invented curve')
    output.mkdir(parents=True,exist_ok=True)
    elapsed=[row['elapsed_seconds'] for row in rows]
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    axes[0].plot(elapsed,[row['process_tree_rss_bytes']/1024**3 for row in rows],color='#156c89')
    axes[0].axhline(12,color='#ac6236',ls='--',label='12 GiB process guard')
    axes[0].axhline(report['physical_memory_bytes']/1024**3,color='#7d8b99',ls=':',label='Physical memory')
    axes[0].set_ylabel('Process-tree RSS (GiB)')
    axes[0].legend(fontsize=8,loc='upper right')
    axes[1].plot(elapsed,[row['system_available_bytes']/1024**3 for row in rows],color='#477c50')
    axes[1].axhline(.5,color='#ac6236',ls='--')
    axes[1].set_ylabel('System available (GiB)')
    counters=[row.get('root_process_pageins_sampled_max') for row in rows]
    if all(value is not None for value in counters):
        axes[2].plot(elapsed,counters,color='#75538b')
    else:
        axes[2].text(.5,.5,'Page-in counters unavailable',ha='center',transform=axes[2].transAxes)
    axes[2].set_ylabel('Root page-ins (sampled)')
    axes[2].set_xlabel('Elapsed runtime seconds (full-file checksum precedes this clock)')
    first=report['time_to_first_nonwhitespace_stdout_seconds']
    for ax in axes:
        if first is not None:
            ax.axvline(first,color='#555',ls=':',lw=1)
        ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(report['model']['filename'].removesuffix('.gguf')+' — local smoke run',fontsize=11)
    fig.text(.5,.015,'Half-second samples. Page-ins are counters, not SSD bytes. Other apps affect system readings; OS file cache is uncontrolled.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.95])
    fig.savefig(output/'inference-memory.png',dpi=180)
    fig.savefig(output/'inference-memory.svg')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plot(json.loads(args.report.read_text()),args.output)
