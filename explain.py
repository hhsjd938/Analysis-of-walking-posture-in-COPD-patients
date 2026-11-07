import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils.data_processor import FEATURE_COLUMNS, SCHEMA


def explain(model_path, features_path, output):
    bundle = joblib.load(model_path)
    if bundle['schema'] != SCHEMA or bundle['feature_columns'] != list(FEATURE_COLUMNS):
        raise ValueError('Model schema mismatch.')
    features = pd.read_csv(features_path)[list(FEATURE_COLUMNS)]
    if features.empty or not np.isfinite(features.to_numpy()).all():
        raise ValueError('SHAP requires finite, nonempty features.')
    values = np.asarray(shap.TreeExplainer(bundle['model'])(features).values)
    importance = np.abs(values).mean(axis=(0, 2)) if values.ndim == 3 else np.abs(values).mean(axis=0)
    table = pd.DataFrame({'feature': FEATURE_COLUMNS, 'mean_absolute_shap': importance}).sort_values('mean_absolute_shap')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / 'shap_importance.csv', index=False)
    figure, axis = plt.subplots(figsize=(10, 9))
    axis.barh(table['feature'], table['mean_absolute_shap'])
    axis.set_xlabel('Mean absolute SHAP value across recordings and classes')
    axis.set_title('Random forest feature contributions')
    figure.tight_layout()
    figure.savefig(output / 'shap_importance.png', dpi=160)
    plt.close(figure)
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('features')
    parser.add_argument('--output', default='results/shap')
    args = parser.parse_args()
    explain(args.model, args.features, args.output)
