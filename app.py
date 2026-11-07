from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from models.predict import predict
from utils.data_processor import analyze, select_track


app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=10 * 1024 * 1024,
                  MODEL_PATH=str(Path(__file__).parent / 'models' / 'model.pth'))


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/upload')
@app.post('/api/upload')
def upload_file():
    file = request.files.get('file')
    try:
        if file is None or not file.filename or not file.filename.lower().endswith('.csv'):
            raise ValueError('請選擇 CSV 檔案。')
        frames, summary, diagnostics = analyze(select_track(pd.read_csv(file.stream), request.form.get('track_id')))
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        if request.path.startswith('/api/'):
            return jsonify(error=str(error)), 400
        return render_template('index.html', error=str(error)), 400
    try:
        prediction = predict(summary, app.config['MODEL_PATH'])
    except Exception:
        app.logger.exception('Model loading or prediction failed')
        prediction = {'status': 'unavailable', 'message': '模型無法載入，請檢查訓練產物；以下顯示步態特徵。'}
    payload = {'features': summary, 'diagnostics': diagnostics, 'frames': len(frames), 'prediction': prediction}
    if request.path.startswith('/api/'):
        return jsonify(payload)
    return render_template('result.html', **payload)


@app.errorhandler(413)
def too_large(error):
    if request.path.startswith('/api/'):
        return jsonify(error='檔案大小上限為 10 MB。'), 413
    return render_template('index.html', error='檔案大小上限為 10 MB。'), 413


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False)
