from pathlib import Path

import numpy as np
import torch

from models.analyzer import SeverityClassifier
from utils.data_processor import FEATURE_COLUMNS, SCHEMA


def predict(summary, path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return {'status': 'unavailable', 'message': '尚未提供訓練完成的模型；以下顯示步態特徵。'}
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('schema') != SCHEMA or bundle.get('feature_columns') != list(FEATURE_COLUMNS):
        raise ValueError('模型特徵格式不符，請用目前的 train.py 重新訓練。')
    mean, scale = np.array(bundle['mean']), np.array(bundle['scale'])
    if mean.shape != (len(FEATURE_COLUMNS),) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('模型標準化參數無效。')
    labels = bundle['labels']
    model = SeverityClassifier(len(FEATURE_COLUMNS), len(labels))
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    values = (np.array([summary[name] for name in FEATURE_COLUMNS]) - mean) / scale
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor(values[None], dtype=torch.float32)), dim=1)[0].tolist()
    selected = int(np.argmax(probabilities))
    accepted = probabilities[selected] >= bundle.get('threshold', 0.0)
    return {'status': 'predicted' if accepted else 'abstained',
            'label': labels[selected] if accepted else None,
            'probabilities': dict(zip(labels, probabilities)),
            'message': '研究分類結果' if accepted else '未達設定信心門檻，保留為未分類。'}
