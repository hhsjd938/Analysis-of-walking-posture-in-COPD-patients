import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.analyzer import SeverityClassifier
from utils.data_processor import analyze, FEATURE_COLUMNS, SCHEMA


def load_dataset(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = ['path', 'subject_id', 'label']
    if not set(required).issubset(manifest.columns) or manifest.empty:
        raise ValueError('Manifest requires path, subject_id, label and at least one recording.')
    if manifest[required].eq('').any().any() or manifest['path'].duplicated().any():
        raise ValueError('Manifest fields must be nonempty and recording paths unique.')
    if manifest.groupby('subject_id')['label'].nunique().max() != 1:
        raise ValueError('Each subject must have one consistent label.')
    rows, resolved_paths = [], set()
    for record in manifest.itertuples():
        path = (manifest_path.parent / record.path).resolve()
        if path in resolved_paths:
            raise ValueError('Two manifest paths resolve to the same recording.')
        resolved_paths.add(path)
        _, summary, _ = analyze(pd.read_csv(path))
        rows.append([summary[name] for name in FEATURE_COLUMNS])
    return manifest, np.asarray(rows, dtype=np.float32)


def split_subjects(manifest, test_size=0.25, seed=42):
    subjects = manifest[['subject_id', 'label']].drop_duplicates()
    if subjects['subject_id'].duplicated().any():
        raise ValueError('Each subject must have one consistent label.')
    if subjects['label'].nunique() < 2 or subjects['label'].value_counts().min() < 2:
        raise ValueError('Need at least two classes and two subjects per class.')
    train_ids, test_ids = train_test_split(subjects['subject_id'], test_size=test_size,
                                          stratify=subjects['label'], random_state=seed)
    return manifest['subject_id'].isin(train_ids).to_numpy(), manifest['subject_id'].isin(test_ids).to_numpy()


def score(y, probabilities, labels, threshold):
    prediction = probabilities.argmax(axis=1)
    accepted = probabilities.max(axis=1) >= threshold
    return {
        'accuracy': float(accuracy_score(y, prediction)),
        'report': classification_report(y, prediction, labels=list(range(len(labels))), target_names=labels,
                                        output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y, prediction, labels=list(range(len(labels)))).tolist(),
        'threshold': threshold, 'coverage': float(accepted.mean()), 'accepted_recordings': int(accepted.sum()),
        'accepted_accuracy': float(accuracy_score(y[accepted], prediction[accepted])) if accepted.any() else None,
    }


def train(manifest_path, output='artifacts', epochs=100, seed=42, test_size=0.25, threshold=0.0):
    if epochs < 1 or not 0 <= threshold <= 1 or not 0 < test_size < 1:
        raise ValueError('Invalid epochs, threshold or test_size.')
    torch.manual_seed(seed)
    np.random.seed(seed)
    manifest, values = load_dataset(manifest_path)
    train_mask, test_mask = split_subjects(manifest, test_size, seed)
    labels = sorted(manifest['label'].unique().tolist())
    y = np.array([labels.index(value) for value in manifest['label']])
    scaler = StandardScaler().fit(values[train_mask])
    x_train = torch.tensor(scaler.transform(values[train_mask]), dtype=torch.float32)
    y_train = torch.tensor(y[train_mask], dtype=torch.long)
    x_test = torch.tensor(scaler.transform(values[test_mask]), dtype=torch.float32)
    model = SeverityClassifier(len(FEATURE_COLUMNS), len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(x_train))
        total = 0.0
        for indices in order.split(128):
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[indices]), y_train[indices])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        losses.append(total / len(x_train))
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(x_test), dim=1).numpy()
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed)
    rf.fit(values[train_mask], y[train_mask])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'schema': SCHEMA, 'feature_columns': list(FEATURE_COLUMNS),
                'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'labels': labels,
                'threshold': threshold}, output / 'model.pth')
    joblib.dump({'model': rf, 'schema': SCHEMA, 'feature_columns': list(FEATURE_COLUMNS), 'labels': labels},
                output / 'random_forest.joblib')
    pd.DataFrame(values[test_mask], columns=FEATURE_COLUMNS).to_csv(output / 'test_features.csv', index=False)
    assignments = manifest[['subject_id', 'label']].copy()
    assignments['split'] = np.where(train_mask, 'train', 'test')
    assignments.to_csv(output / 'split.csv', index=False)
    metrics = {'schema': SCHEMA, 'seed': seed, 'epochs': epochs, 'labels': labels,
               'train_subjects': int(manifest.loc[train_mask, 'subject_id'].nunique()),
               'test_subjects': int(manifest.loc[test_mask, 'subject_id'].nunique()),
               'train_recordings': int(train_mask.sum()), 'test_recordings': int(test_mask.sum()),
               'mlp': score(y[test_mask], probabilities, labels, threshold),
               'random_forest': score(y[test_mask], rf.predict_proba(values[test_mask]), labels, threshold),
               'training_loss': losses}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--threshold', type=float, default=0.0)
    args = parser.parse_args()
    train(args.manifest, args.output, args.epochs, args.seed, args.test_size, args.threshold)
