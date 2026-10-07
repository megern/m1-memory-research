"""Plot separate time, MLX allocation and sampled RSS metrics; no combined RAM claim."""
import argparse
import json
from pathlib import Path


def plot(summary, output):
    import os
    cache = Path.cwd()/'work/matplotlib-cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data = json.loads(summary.read_text())
    modes = list(data['modes'])
    label_map = {'resident': 'Resident', 'file': 'File sync', 'bytes': 'Bytes sync',
                 'prefetch1': 'Prefetch 1', 'prefetch2': 'Prefetch 2',
                 'reuse': 'Reuse only', 'cache': 'Cache 7 layers'}
    labels = [label_map[m] for m in modes]
    rows = [r for r in data['trials'] if r['exit_code'] == 0 and r['guard'] is None
            and r.get('logits_allclose') and r.get('tokens_all_equal')]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5))
    color_map = dict(zip(['resident', 'file', 'bytes', 'prefetch1', 'prefetch2', 'reuse', 'cache'],
                         ['#596579', '#176f62', '#7476ad', '#d78f38', '#ba594f', '#7476ad', '#d78f38']))
    colors = [color_map[m] for m in modes]
    metrics = [('forward_seconds', 1, 'Forward time (seconds)'),
               ('mlx_peak_allocated_bytes', 1024**2, 'Peak MLX allocation (MiB)'),
               ('sampled_process_tree_rss_peak_bytes', 1024**2, 'Sampled process-tree RSS (MiB)')]
    for ax, (key, divisor, title) in zip(axes, metrics):
        values = [[r[key]/divisor for r in rows if r['mode'] == mode] for mode in modes]
        box = ax.boxplot(values, tick_labels=labels, patch_artist=True, widths=.55,
                         medianprops={'color': '#152536', 'linewidth': 1.5}, showfliers=False)
        for item, color in zip(box['boxes'], colors):
            item.set_facecolor(color)
            item.set_alpha(.35)
        for index, (points, color) in enumerate(zip(values, colors), start=1):
            ax.scatter(index+np.linspace(-.1,.1,len(points)), points, color=color, s=23, zorder=3)
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(axis='x', labelrotation=35, labelsize=8)
        ax.grid(axis='y', alpha=.22)
        ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False)
    kind = 'bounded cache' if 'cache' in modes else 'scheduling'
    reference = json.loads((summary.parent/'reference.json').read_text())
    count = len(reference['cases'])
    steps = sum(len(c['tokens']) for c in reference['cases'])
    rounds = data['modes'][modes[0]]['planned_trials']
    fig.suptitle(f'Original BF16 weights: {kind} controls on M1 / 16 GiB', fontsize=13, y=.98)
    fig.text(.5,.02, f'{rounds} planned trials per mode; {count} short prompts / {steps} steps per successful trial. Warmed, uncontrolled OS cache.\n'
             'MLX allocation and RSS overlap and use different accounting; do not add them. No cold-SSD or 70B claim.',
             ha='center', fontsize=8, color='#43505d')
    fig.tight_layout(rect=(0,.12,1,.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=240)
    fig.savefig(output.with_suffix('.svg'))
    svg = output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    plot(a.summary, a.output)
