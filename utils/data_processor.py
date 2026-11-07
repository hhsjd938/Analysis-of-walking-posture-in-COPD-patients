import json

import numpy as np
import pandas as pd


SIGNALS = (
    'head_torso_angle', 'left_elbow_angle', 'right_elbow_angle',
    'left_hip_angle', 'right_hip_angle', 'trunk_tilt',
    'ankle_horizontal_separation', 'shoulder_hip_distance',
    'head_hip_distance', 'torso_midpoint_x', 'torso_midpoint_y',
)
FEATURE_COLUMNS = tuple(f'{stat}_{name}' for name in SIGNALS for stat in ('mean', 'std'))
SCHEMA = 'coco17_xy_confidence_v1'


def select_track(df, track_id=None):
    if track_id is None or str(track_id).strip() == '':
        return df
    if 'id' not in df:
        raise ValueError('指定追蹤 ID 時，CSV 必須有 id 欄位。')
    selected = df.loc[df['id'].astype(str) == str(track_id).strip()].copy()
    if selected.empty:
        raise ValueError('CSV 中找不到指定的追蹤 ID。')
    return selected


def parse_keypoints(df):
    if df.empty:
        raise ValueError('CSV 沒有資料列。')
    if len(df) > 100000:
        raise ValueError('每次最多處理 100000 列。')
    if 'id' in df and df['id'].nunique(dropna=False) != 1:
        raise ValueError('一份 CSV 必須只包含同一位追蹤對象。')
    columns = [f'kpt_{i}' for i in range(1, 52)]
    if all(name in df for name in columns):
        values = df[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    elif 'kpts' in df:
        rows = []
        for value in df['kpts']:
            text = str(value).strip().strip('"')
            try:
                row = np.asarray(json.loads(text if text.startswith('[') else '[' + text + ']'), dtype=float).ravel()
            except (ValueError, TypeError) as error:
                raise ValueError('kpts 必須包含 17 組 x、y、confidence 數值。') from error
            if row.size != 51:
                raise ValueError('每列 kpts 必須恰好包含 51 個數值。')
            rows.append(row)
        values = np.asarray(rows)
    else:
        raise ValueError('缺少 kpts 欄位或完整的 kpt_1 到 kpt_51。')
    if not np.isfinite(values).all():
        raise ValueError('座標與信心分數必須是有限數值。')
    points = values.reshape(-1, 17, 3)
    if ((points[:, :, 2] < 0) | (points[:, :, 2] > 1)).any():
        raise ValueError('confidence 必須介於 0 與 1。')
    return points


def joint_angle(a, b, c):
    u, v = a - b, c - b
    norm = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cosine = np.divide(np.sum(u * v, axis=1), norm, out=np.full(len(norm), np.nan), where=norm > 0)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def analyze(df, min_confidence=0.25):
    if not 0 <= min_confidence <= 1:
        raise ValueError('min_confidence 必須介於 0 與 1。')
    points = parse_keypoints(df)
    xy = points[:, :, :2].copy()
    xy[points[:, :, 2] < min_confidence] = np.nan
    nose = xy[:, 0]
    shoulders = (xy[:, 5] + xy[:, 6]) / 2
    hips = (xy[:, 11] + xy[:, 12]) / 2
    midpoint = (shoulders + hips) / 2
    trunk = shoulders - hips
    tilt = np.degrees(np.arctan2(trunk[:, 0], -trunk[:, 1]))
    tilt[np.linalg.norm(trunk, axis=1) == 0] = np.nan
    frames = pd.DataFrame({
        'head_torso_angle': joint_angle(nose, shoulders, hips),
        'left_elbow_angle': joint_angle(xy[:, 5], xy[:, 7], xy[:, 9]),
        'right_elbow_angle': joint_angle(xy[:, 6], xy[:, 8], xy[:, 10]),
        'left_hip_angle': joint_angle(xy[:, 5], xy[:, 11], xy[:, 13]),
        'right_hip_angle': joint_angle(xy[:, 6], xy[:, 12], xy[:, 14]),
        'trunk_tilt': tilt,
        'ankle_horizontal_separation': np.abs(xy[:, 15, 0] - xy[:, 16, 0]),
        'shoulder_hip_distance': np.linalg.norm(shoulders - hips, axis=1),
        'head_hip_distance': np.linalg.norm(nose - hips, axis=1),
        'torso_midpoint_x': midpoint[:, 0],
        'torso_midpoint_y': midpoint[:, 1],
    })
    if frames.notna().sum().eq(0).any():
        raise ValueError('有特徵完全沒有有效關鍵點；請檢查座標配置及信心分數。')
    summary, diagnostics = {}, {}
    for name in SIGNALS:
        series = frames[name].dropna()
        q1, q2, q3 = series.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        summary[f'mean_{name}'] = float(series.mean())
        summary[f'std_{name}'] = float(series.std(ddof=0))
        outliers = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        diagnostics[name] = {'valid_frames': int(len(series)), 'q1': float(q1), 'median': float(q2),
                             'q3': float(q3), 'iqr': float(iqr), 'range': float(series.max() - series.min()),
                             'iqr_outliers': int(outliers.sum())}
    return frames, summary, diagnostics


def extract_features_from_rows(df, file_name=None, label=None):
    frames, summary, diagnostics = analyze(df)
    return {'row_features': frames.astype(object).where(frames.notna(), None).to_dict('records'),
            'statistical_features': summary, 'diagnostics': diagnostics}
