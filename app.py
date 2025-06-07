from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import pandas as pd
import torch
from werkzeug.utils import secure_filename
from utils.data_processor import extract_features_from_rows 
from models.analyzer import SeverityClassifier

# 配置
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'csv'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 確保上傳文件夾存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 確保所需目錄存在
required_directories = ['./uploads', './static/js', './static/css', './templates', './models', './utils', './results']
for directory in required_directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 處理文件並運行分析
        df = pd.read_csv(file_path)
        features = extract_features_from_rows(df, filename, label=1)  # 臨時標籤用於演示
        statistical_features = features["statistical_features"]

        # 運行模型預測
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SeverityClassifier(input_size=len(statistical_features), num_classes=4).to(device)
        model.load_state_dict(torch.load("models/model.pth", map_location=device))
        model.eval()
        
        # 準備模型數據
        feature_values = list(statistical_features.values())[:-2]  # 排除 'label' 和 'file_name'
        feature_tensor = torch.tensor(feature_values, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(feature_tensor)
            _, predicted_class = torch.max(output.data, 0)
        
        # 返回結果給用戶
        return render_template('result.html', result=int(predicted_class), features=statistical_features)
    else:
        return "File type not allowed", 400

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/upload', methods=['POST'])
def api_upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully.", "filename": filename}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == "__main__":
    app.run(debug=True)