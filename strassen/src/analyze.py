"""Validate complete runs and derive the required tables without cherry-picking."""
from run import write_json
import argparse
import csv
import json
from pathlib import Path
import statistics


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, repeats, processes, entries):
    expected = {(r, p, e['group'], e['pair']) for r in range(repeats)
                for p in processes for e in entries}
    indexed = {(x['repeat'], x['processes'], x['group'], x['pair']): x for x in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('missing, duplicate, or unexpected measurements')
    for e in entries:
        hashes = {indexed[r, p, e['group'], e['pair']]['output_sha256']
                  for r in range(repeats) for p in processes}
        if len(hashes) != 1:
            raise ValueError('inconsistent outputs')
    if any(not x['verified'] or not (0 < x['seconds'] < float('inf')) for x in rows):
        raise ValueError('invalid timing/verification')
    totals = {p: [sum(x['seconds'] for x in rows if x['repeat'] == r and x['processes'] == p)
                  for r in range(repeats)] for p in processes}
    # One whole repetition per mode, ranked by its complete workload time.
    selected = {p: sorted(range(repeats), key=lambda r: totals[p][r])[repeats // 2] for p in processes}
    stability = [dict(processes=p, selected_repeat=selected[p],
                      total_min_s=min(totals[p]), total_median_s=statistics.median(totals[p]),
                      total_max_s=max(totals[p]),
                      total_mean_s=statistics.mean(totals[p]),
                      total_cv=statistics.stdev(totals[p]) / statistics.mean(totals[p]) if repeats > 1 else None)
                 for p in processes]
    return indexed, selected, stability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    args = parser.parse_args()
    root = Path(args.experiment)
    if not (root / 'complete.json').exists():
        raise SystemExit('experiment is incomplete')
    config = json.loads((root / 'configuration.json').read_text())
    manifest = json.loads((root / 'input_manifest.json').read_text())
    entries = sorted([e for e in manifest['entries'] if e['group'] in config['groups'] and e['size'] <= config['max_size']],
                     key=lambda e: (e['group'], e['pair']))
    rows = [json.loads(line) for line in (root / 'measurements.jsonl').read_text().splitlines()]
    processes = sorted(config['processes'])
    if 0 not in processes or not any(p > 0 for p in processes) or 0 not in config['groups']:
        raise SystemExit('CA2 analysis requires sequential, at least one MP mode, and G0 results')
    indexed, selected, stability = summarize(rows, config['repeats'], processes, entries)
    dest = root / 'derived'
    dest.mkdir(exist_ok=True)
    if config['repeats'] > 1:
        write_csv(dest / 'repeat_stability.csv', stability)
        write_json(dest / 'selected_repetitions.json', selected)
    pair_times, group_times = {}, {}
    groups = sorted(set(e['group'] for e in entries))
    for p in processes:
        pair_times[p] = []
        group_times[p] = []
        cumulative_groups = 0.0
        for g in groups:
            cumulative = 0.0
            for e in (e for e in entries if e['group'] == g):
                previous = cumulative
                cumulative += indexed[selected[p], p, g, e['pair']]['seconds']
                if g == 0:
                    pair_times[p].append(dict(pair=e['pair'], size=e['size'], cumulative_s=cumulative,
                                              derived_pair_s=cumulative - previous))
            cumulative_groups += cumulative
            group_times[p].append(dict(group=g, total_s=cumulative, cumulative_s=cumulative_groups))
    write_csv(dest / 'B-1.1.csv', pair_times[0])
    write_csv(dest / 'B-1.2.csv', group_times[0])
    mp_counts = [p for p in processes if p]
    c1, c2 = [], []
    for i, base in enumerate(pair_times[0]):
        row = dict(pair=base['pair'], size=base['size'])
        for p in mp_counts:
            row[f'p{p}_cumulative_s'] = pair_times[p][i]['cumulative_s']
        for p in mp_counts:
            row[f'p{p}_derived_pair_s'] = pair_times[p][i]['derived_pair_s']
        c1.append(row)
    for i, g in enumerate(groups):
        row = dict(group=g)
        for p in mp_counts:
            row[f'p{p}_total_s'] = group_times[p][i]['total_s']
        for p in mp_counts:
            row[f'p{p}_cumulative_s'] = group_times[p][i]['cumulative_s']
        c2.append(row)
    write_csv(dest / 'C-1.1.csv', c1)
    write_csv(dest / 'C-1.2.csv', c2)
    d1 = [dict(processes=p, completed_pairs=i + 1, largest_size=b['size'],
               speedup=b['cumulative_s'] / pair_times[p][i]['cumulative_s'])
          for p in mp_counts for i, b in enumerate(pair_times[0])]
    d2 = [dict(processes=p, through_group=g,
               speedup=group_times[0][i]['cumulative_s'] / group_times[p][i]['cumulative_s'])
          for p in mp_counts for i, g in enumerate(groups)]
    write_csv(dest / 'D-1.1.csv', d1)
    write_csv(dest / 'D-1.2.csv', d2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 160, 'axes.grid': True, 'grid.alpha': .25})
    caveats = []
    if (root / 'load_observation.json').exists():
        caveats.append('Concurrent user activity observed')

    def save_figure(fig, filename):
        if caveats:
            fig.text(.5, .025, ' | '.join(caveats), ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .065 if caveats else 0, 1, 1))
        fig.savefig(dest / filename)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    complete = [x for x in d1 if x['completed_pairs'] == len(pair_times[0])]
    workload_label = 'complete G0' if len(pair_times[0]) == 8 else f'partial G0 ({len(pair_times[0])} pairs)'
    ax.plot(mp_counts, [x['speedup'] for x in complete], 'o-', label=f'Measured, {workload_label}')
    ax.plot(mp_counts, mp_counts, '--', color='gray', label='Ideal linear speed-up')
    ax.axhline(1, color='gray', linewidth=.8)
    ax.set(xlabel='Worker processes', ylabel='Sequential time / MP time', title=f'Strassen: {workload_label} speed-up', xticks=mp_counts)
    ax.legend()
    save_figure(fig, 'D-1.1_processes.png')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in mp_counts:
        series = [x for x in d1 if x['processes'] == p]
        ax.plot([x['completed_pairs'] for x in series], [x['speedup'] for x in series], 'o-', label=f'p={p}')
    ax.axhline(1, color='gray', linewidth=.8)
    ax.set(xlabel='Number of completed pairs (ascending matrix size)', ylabel='Cumulative speed-up', title='Strassen: G0 cumulative workload', xticks=range(1, len(pair_times[0]) + 1))
    ax.legend(ncol=3)
    save_figure(fig, 'D-1.1_workload.png')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for g in groups:
        ax.plot(mp_counts, [x['speedup'] for x in d2 if x['through_group'] == g], 'o-', label=f'G0 through G{g}')
    ax.axhline(1, color='gray', linewidth=.8)
    ax.set(xlabel='Worker processes', ylabel='Cumulative speed-up', title='Strassen: cumulative group workloads', xticks=mp_counts)
    ax.legend()
    save_figure(fig, 'D-1.2_groups.png')
    print(json.dumps(stability, indent=2))


if __name__ == '__main__':
    main()
