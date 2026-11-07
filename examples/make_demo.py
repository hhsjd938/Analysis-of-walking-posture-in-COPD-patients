import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_recording(frames=60, amplitude=0.1, phase=0.0):
    base = np.array([[0, 0], [-.1, -.05], [.1, -.05], [-.2, 0], [.2, 0],
                     [-.4, .7], [.4, .7], [-.5, 1.2], [.5, 1.2], [-.6, 1.7], [.6, 1.7],
                     [-.3, 1.8], [.3, 1.8], [-.3, 2.5], [.3, 2.5], [-.3, 3.2], [.3, 3.2]])
    rows = []
    for frame in range(frames):
        xy = base.copy()
        offset = amplitude * np.sin(frame / 6 + phase)
        xy[[7, 9, 13, 15], 0] += offset
        xy[[8, 10, 14, 16], 0] -= offset
        xy[:, 0] += frame * .02
        values = np.column_stack([xy, np.full(17, .99)]).ravel().tolist()
        rows.append({'kpts': json.dumps(values)})
    return pd.DataFrame(rows)


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    make_recording().to_csv(output / 'synthetic_pose.csv', index=False)
    manifest = []
    for label in range(4):
        for subject in range(4):
            name = f'synthetic_{label}_{subject}.csv'
            make_recording(amplitude=.05 + .05 * label, phase=subject / 3).to_csv(output / name, index=False)
            manifest.append({'path': name, 'subject_id': f'synthetic-{label}-{subject}', 'label': f'demo-{label}'})
    pd.DataFrame(manifest).to_csv(output / 'manifest.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/demo')
    args = parser.parse_args()
    generate(args.output)
