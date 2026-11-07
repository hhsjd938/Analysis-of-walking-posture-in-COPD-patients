import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app import app
from examples.make_demo import generate, make_recording
from models.predict import predict
from train import load_dataset, score, split_subjects, train
from utils.data_processor import analyze, joint_angle, parse_keypoints, select_track, FEATURE_COLUMNS


class FeatureTests(unittest.TestCase):
    def test_known_geometry(self):
        a = np.array([[0., 0.], [0., 0.], [0., 0.]])
        b = np.array([[1., 0.], [1., 0.], [0., 0.]])
        c = np.array([[1., 1.], [2., 0.], [1., 1.]])
        result = joint_angle(a, b, c)
        np.testing.assert_allclose(result[:2], [90., 180.])
        self.assertTrue(np.isnan(result[2]))

    def test_triplet_mapping_and_statistics(self):
        frames, summary, diagnostics = analyze(make_recording(frames=4, amplitude=0))
        np.testing.assert_allclose(frames['ankle_horizontal_separation'], .6)
        np.testing.assert_allclose(frames['head_torso_angle'], 180.)
        np.testing.assert_allclose(frames['trunk_tilt'], 0.)
        self.assertAlmostEqual(summary['mean_shoulder_hip_distance'], 1.1)
        self.assertEqual(diagnostics['ankle_horizontal_separation']['iqr_outliers'], 0)
        self.assertEqual(list(summary), list(FEATURE_COLUMNS))

    def test_expanded_format_matches_kpts(self):
        source = make_recording(frames=4)
        points = parse_keypoints(source)
        expanded = pd.DataFrame(points.reshape(4, 51), columns=[f'kpt_{i}' for i in range(1, 52)])
        np.testing.assert_allclose(analyze(source)[0], analyze(expanded)[0])

    def test_invalid_inputs(self):
        for df in (pd.DataFrame(), pd.DataFrame({'kpts':['1,2,3']}), pd.DataFrame({'other':[1]}),
                   pd.DataFrame({'kpts':['x']})):
            with self.subTest(df=df), self.assertRaises(ValueError):
                analyze(df)
        df = make_recording(frames=2)
        df['id'] = [1, 2]
        with self.assertRaises(ValueError):
            analyze(df)

    def test_low_confidence_is_missing_not_zero(self):
        df = make_recording(frames=3)
        values = json.loads(df.loc[0, 'kpts'])
        values[15 * 3 + 2] = .01
        df.loc[0, 'kpts'] = json.dumps(values)
        frames, _, diagnostics = analyze(df)
        self.assertTrue(np.isnan(frames.loc[0, 'ankle_horizontal_separation']))
        self.assertEqual(diagnostics['ankle_horizontal_separation']['valid_frames'], 2)
        with self.assertRaises(ValueError):
            analyze(df.iloc[:1])

    def test_nonfinite_and_invalid_confidence(self):
        for value in (float('nan'), 1.1, -.1):
            df = make_recording(frames=1)
            points = json.loads(df.loc[0, 'kpts'])
            points[2] = value
            df.loc[0, 'kpts'] = json.dumps(points)
            with self.subTest(value=value), self.assertRaises(ValueError):
                analyze(df)

    def test_explicit_track_selection(self):
        df = make_recording(frames=4)
        df['id'] = [1, 2, 1, 2]
        selected = select_track(df, '2')
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected.index.tolist(), [1, 3])
        self.assertEqual(len(analyze(selected)[0]), 2)
        with self.assertRaises(ValueError):
            select_track(df, '3')


class ModelTests(unittest.TestCase):
    def test_subjects_never_cross_split(self):
        records = [{'subject_id':f'{label}-{subject}', 'label':str(label)}
                   for label in range(4) for subject in range(4) for _ in range(2)]
        manifest = pd.DataFrame(records)
        train_mask, test_mask = split_subjects(manifest)
        self.assertFalse(set(manifest.loc[train_mask, 'subject_id']) & set(manifest.loc[test_mask, 'subject_id']))
        self.assertEqual(set(manifest.loc[test_mask, 'label']), {'0','1','2','3'})

    def test_threshold_reports_rejections(self):
        result = score(np.array([0, 1]), np.array([[.9,.1],[.6,.4]]), ['a','b'], .8)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['coverage'], .5)
        self.assertEqual(result['accepted_accuracy'], 1.)
        self.assertIsNone(score(np.array([0]), np.array([[.5,.5]]), ['a','b'], 1.)['accepted_accuracy'])

    def test_training_checkpoint_roundtrip_and_manifest_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            generate(directory / 'data')
            output = directory / 'artifacts'
            metrics = train(directory / 'data/manifest.csv', output=output, epochs=1)
            self.assertEqual(metrics['test_subjects'], 4)
            _, summary, _ = analyze(make_recording())
            prediction = predict(summary, output / 'model.pth')
            self.assertEqual(prediction['status'], 'predicted')
            self.assertAlmostEqual(sum(prediction['probabilities'].values()), 1., places=5)
            bundle = torch.load(output / 'model.pth', weights_only=True)
            manifest, values = load_dataset(directory / 'data/manifest.csv')
            train_mask, _ = split_subjects(manifest)
            np.testing.assert_allclose(bundle['mean'], values[train_mask].astype(float).mean(axis=0), atol=1e-6)
            bundle['schema'] = 'incompatible'
            torch.save(bundle, output / 'model.pth')
            with self.assertRaises(ValueError):
                predict(summary, output / 'model.pth')
            manifest.loc[1, 'path'] = manifest.loc[0, 'path']
            manifest.to_csv(directory / 'data/manifest.csv', index=False)
            with self.assertRaises(ValueError):
                load_dataset(directory / 'data/manifest.csv')


class WebTests(unittest.TestCase):
    def setUp(self):
        self.previous = app.config['MODEL_PATH']
        app.config.update(TESTING=True, MODEL_PATH='models/not-present.pth')
        self.client = app.test_client()

    def tearDown(self):
        app.config['MODEL_PATH'] = self.previous

    def upload(self, path, content=None):
        content = content if content is not None else make_recording(frames=5).to_csv(index=False).encode()
        return self.client.post(path, data={'file':(io.BytesIO(content), 'pose.csv')})

    def test_web_without_weights(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        response = self.upload('/upload')
        self.assertEqual(response.status_code, 200)
        self.assertIn('尚未提供訓練完成的模型', response.get_data(as_text=True))

    def test_api_without_weights(self):
        response = self.upload('/api/upload')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['prediction']['status'], 'unavailable')
        self.assertEqual(response.json['frames'], 5)

    def test_api_explicit_track(self):
        df = make_recording(frames=4)
        df['id'] = [1, 2, 1, 2]
        response = self.client.post('/api/upload', data={'file':(io.BytesIO(df.to_csv(index=False).encode()), 'pose.csv'), 'track_id':'2'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['frames'], 2)

    def test_invalid_upload_and_private_route(self):
        self.assertEqual(self.client.post('/api/upload').status_code, 400)
        self.assertEqual(self.upload('/api/upload', b'bad\n1\n').status_code, 400)
        self.assertEqual(self.client.get('/results/pose.csv').status_code, 404)

    def test_upload_limit(self):
        previous = app.config['MAX_CONTENT_LENGTH']
        try:
            app.config['MAX_CONTENT_LENGTH'] = 10
            self.assertEqual(self.upload('/api/upload').status_code, 413)
        finally:
            app.config['MAX_CONTENT_LENGTH'] = previous


if __name__ == '__main__':
    unittest.main()
