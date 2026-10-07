"""Make exportable scientific figures from the recorded all-seed summary."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'runs/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for row, task in enumerate(['auth', 'workflow']):
        data = summary['tasks'][task]
        for column, (scoring, title) in enumerate([
                ('strict', 'Strict JSON classification'),
                ('outer_fence_allowed', 'Classification allowing only outer JSON fence')]):
            ax = axes[row, column]
            entries = data['entries']
            values = [e[scoring]['accuracy']*100 for e in entries]
            bars = ax.bar(range(len(entries)), values,
                          color=['#7d8b99', '#156c89', '#156c89', '#156c89'], width=.62)
            ax.axhline(data['majority_accuracy']*100, color='#ac6236', ls='--', lw=1,
                       label='Majority')
            ax.set_xticks(range(len(entries)), ['Base', 'Seed 17', 'Seed 23', 'Seed 41'])
            ax.set_ylim(0, 112)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_title(f'{task.capitalize()}: {title}', fontsize=10)
            ax.set_ylabel('Accuracy (%)')
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+2, f'{value:.1f}',
                        ha='center', va='bottom', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.15)
            ax.set_axisbelow(True)
            if column == 0:
                ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Local Qwen3 0.6B LoRA: frozen synthetic tests, all training seeds', fontsize=13)
    fig.text(.5, .015, '96 cases per task. Stated rules and shared templates limit generalization; these are not operational security benchmarks.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .035, 1, .95])
    fig.savefig(output/'specialist-accuracy.png', dpi=180)
    fig.savefig(output/'specialist-accuracy.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text()), args.output)
