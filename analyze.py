import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_processor import analyze, select_track, SIGNALS


def export_analysis(csv_path, output, track_id=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames, summary, diagnostics = analyze(select_track(pd.read_csv(csv_path), track_id))
    frames.to_csv(output / 'frame_features.csv', index=False)
    pd.DataFrame([summary]).to_csv(output / 'recording_features.csv', index=False)
    (output / 'quartiles.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    figure, axes = plt.subplots(4, 3, figsize=(15, 12))
    for axis, name in zip(axes.flat, SIGNALS):
        axis.boxplot(frames[name].dropna())
        axis.set_title(name, fontsize=9)
    axes.flat[-1].axis('off')
    figure.tight_layout()
    figure.savefig(output / 'feature_boxplots.png', dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(3, 1, figsize=(12, 8))
    for axis, name in zip(axes, ('left_elbow_angle', 'left_hip_angle', 'ankle_horizontal_separation')):
        axis.plot(frames.index, frames[name])
        axis.set_ylabel(name.replace('_', ' '))
        axis.set_xlabel('Row order (not elapsed time)')
    figure.tight_layout()
    figure.savefig(output / 'feature_traces.png', dpi=160)
    plt.close(figure)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--output', default='results/analysis')
    parser.add_argument('--track-id')
    args = parser.parse_args()
    export_analysis(args.csv, args.output, args.track_id)
