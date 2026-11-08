# 步行姿態於 COPD 病患分析

**亞洲大學資訊工程學系｜113 學年度畢業專題**

本專題以臺中榮民總醫院提供的 COPD 病患步態影像為研究資料，透過 **YOLOv7 人體關鍵點擷取、步態特徵分析與多層感知機（MLP）**，探索病患由輕至重四類病情的分類，並以 SHAP 分析特徵貢獻，作為活動能力量化評估的研究。

依專題海報記載，資料涵蓋 **71 名 COPD 病患、約 60,000 筆資料點**。研究重點是從行走姿態中找出可供分類的特徵，結合統計圖表、模型評估及結果展示。

[English](#english) · [研究資料對照](docs/report_alignment.md) · [程式操作筆記本](notebooks/gait_workflow.ipynb) · [資料格式](docs/data_format.md)

## 研究流程

1. **影片與關鍵點擷取：** 將步態影片轉為逐幀圖片，使用 YOLOv7 提取人體關鍵點並輸出 CSV。
2. **資料前處理：** 整理錯誤辨識與缺失資料，拆解 KPTS 座標欄位，並使用 StandardScaler 進行特徵標準化。
3. **步態特徵分析：** 分析頭部與軀幹角度、手臂擺幅角度、步距／步長、髖關節角度，以及平均值、標準差等統計特徵；以四分位數、盒鬚圖與離群值觀察類別差異。
4. **四類分類：** 建立 PyTorch MLP，預測研究資料中的四類 COPD 病患類別。
5. **結果解釋與呈現：** 使用混淆矩陣、Precision、Recall、F1、ROC／PR 曲線及 SHAP 分析結果，並呈現步態評估介面。

報告第 31–33 頁的架構為 **32 → 64 → 16 的三個隱藏層，加上四類輸出層**，共四層全連接；使用 ReLU、Dropout **0.1**、Adam，學習率 **0.001**。報告文字記載 30 個輸入特徵；完整特徵清單與各次實驗的對應情況見[研究資料對照](docs/report_alignment.md)。

## 專題成果

| 項目 | 原始文件記錄 |
| --- | --- |
| 測試準確率 | 報告第 35 頁截圖為 **79.07%**；同頁文字描述約 **72%–79%** |
| Precision | **0.76** |
| Recall | **0.79** |
| F1 | **0.77** |
| 海報成果摘要 | 準確率 **80%** |

以上依報告及海報分別呈現，是當時的實驗記錄；本次文件整理未重新執行該實驗。

報告中的 SHAP 分析指出，**平均頭部與軀幹角度、肩膀至髖部距離標準差、步長標準差**是主要影響特徵。樣本較多的第一、第二類表現較好，第三、第四類較難分類，顯示類別不平衡及資料量是後續改善重點。後續方向包括擴充病患資料、改善特徵與模型，並延伸結果展示介面。

## 本儲存庫內容

收錄資料前處理、步態統計、MLP 分類與 SHAP 相關程式，包含從 127 份 Colab 筆記本中篩選出的 **30 份步態研究筆記**、原 GitHub 訓練來源，以及可直接操作的 CSV 分析與 Flask 展示流程。[來源盤點](docs/source_inventory.md)／[完整索引](docs/source_index.md)

目前可執行入口從**關鍵點 CSV**開始；影片轉幀及 YOLOv7 擷取屬專題研究流程，未包成此儲存庫的一鍵影片分析功能。研究內容依報告與海報整理；下方操作設定則對應目前儲存庫中的實際程式。

## 專案結構

```text
app.py                       CSV 上傳與結果頁面
analyze.py                   特徵、統計與圖表匯出
train.py                     依受試者分組的 MLP / Random Forest 訓練
explain.py                   Random Forest SHAP
utils/data_processor.py      統一座標解析及特徵定義
models/analyzer.py           256 → 128 → 64 的 MLP
models/predict.py            模型格式檢查與推論
templates/、static/css/       網頁畫面
examples/make_demo.py        合成資料產生器
notebooks/gait_workflow.ipynb 可執行的整合筆記本
research/                   30 份 Colab 筆記及原 GitHub 訓練來源
docs/                       資料格式、來源盤點與示例圖
tests/                      特徵、資料分割、模型與網頁測試
requirements.txt            執行依賴
requirements-explain.txt    SHAP 額外依賴
```

`research/` 保留探索過程，已移除執行輸出與 Colab 中繼資料；其中有獨立實驗及未統一的座標定義，不保證整本依序執行。日常操作請使用根目錄程式或 `notebooks/gait_workflow.ipynb`。

## 快速開始

建議 Python 3.11 或 3.12。在專案根目錄執行；需要隔離環境時先建立並啟用虛擬環境。

```bash
python -m pip install -r requirements.txt
python examples/make_demo.py --output data/demo
python analyze.py data/demo/synthetic_pose.csv --output results/demo
python app.py
```

![合成座標操作示例](docs/images/synthetic_feature_traces.png)

上圖是程式操作的合成資料示例，不是當年病患實驗圖表。

開啟 <http://127.0.0.1:5000>，選擇 `data/demo/synthetic_pose.csv`。分析結果包含逐列特徵 CSV、記錄摘要 CSV、四分位數 JSON、盒鬚圖及折線圖。網頁上傳只在記憶體處理，每次最多 10 MB／100000 列，不提供原始檔下載路由。

也可開啟整合筆記本，依序完成合成資料、分析、訓練煙霧測試及 SHAP；Colab 中會先取得此儲存庫並安裝依賴。

## 使用自己的資料

每份 CSV 必須是**同一段記錄、同一位追蹤對象**，包含 `kpts` 的 51 個數值，或 `kpt_1` 到 `kpt_51`。目前明訂使用 **COCO 17 點、每點依序 x/y/confidence**；來源資料的實際配置需先確認。座標向右、向下增加；不同配置需先轉換。[完整格式與特徵公式](docs/data_format.md)

踝部水平間距是影像座標中的兩踝 x 差，並非經校正的實際步長；軀幹中點並非人體質心。沒有可靠時間或幀率時，不以列號推算步頻、速度、加速度。IQR 離群值保留並標記，低信心關鍵點不補成零。

若 CSV 含多個追蹤 ID，可在網頁填寫要分析的 ID，或使用 `python analyze.py data/recording.csv --track-id 1`。訓練前仍須將不同對象與錄製片段分成獨立檔案；追蹤 ID 不等同跨影片的受試者身分。

分類資料需自行準備 `data/manifest.csv`，欄位如下；相對路徑以 manifest 所在資料夾為準：

```csv
path,subject_id,label
recording_001.csv,subject_001,class_A
recording_002.csv,subject_002,class_B
```

以上兩列只示範格式。訓練需要每類至少兩位不同受試者，且測試比例須容納所有類別；同一人所有記錄須使用相同 `subject_id` 和類別。類別名稱由研究標籤表提供，不會自動解讀為疾病分期或健康／異常。

## 程式訓練與 SHAP

```bash
python train.py data/manifest.csv --output artifacts/run --epochs 100 --test-size 0.25 --seed 42
python -m pip install -r requirements-explain.txt
python explain.py artifacts/run/random_forest.joblib artifacts/run/test_features.csv --output results/shap
```

目前 `train.py` 使用每段記錄的 22 個平均／標準差特徵；以受試者分組並按類別切分，標準化只在訓練集擬合。這個操作入口沿用 GitHub 網頁模型的 256→128→64 隱藏層與 Dropout 0.45，與上方報告的研究架構不同；Random Forest 使用 200 棵樹、最大深度 8。`explain.py` 的 SHAP 解釋對象是 Random Forest；不能據此將當年 MLP 的分類成績歸給 Random Forest。

訓練產生 `model.pth`、`random_forest.joblib`、`metrics.json`、`split.csv` 及 `test_features.csv`。MLP 權重同時保存特徵順序、座標格式、標準化參數與類別名稱。將自己訓練的 `model.pth` 放到 `models/model.pth`，重新啟動網頁即可使用；只載入自己產生或可信任的模型檔。

預設 `--threshold 0` 不拒絕分類。若事先設定較高門檻，評估會同時列出全體準確率、接受樣本準確率與覆蓋率，不隱藏未分類樣本。門檻或參數調整須使用額外驗證集，保留測試集作最終評估。指標以「記錄」計算，同一受試者可能有多段記錄；資料量、類別分布與分割方式仍須在實際研究中交代。

## 驗證與資料狀態

```bash
python -m unittest discover -s tests -v
```

測試涵蓋已知幾何角度、三元座標映射、低信心值、錯誤輸入、受試者不跨集合、訓練／推論格式、閾值覆蓋率及網頁上傳。合成資料可用 `python train.py data/demo/manifest.csv --output artifacts/demo --epochs 2` 做流程煙霧測試，數值不代表研究效能。

研究描述與成果已依提供的專題報告書及海報核對，文件頁碼與程式差異記錄於[研究資料對照](docs/report_alignment.md)。儲存庫提供程式、研究筆記及合成示例，尚未包含可重現當年實驗的完整標籤資料與已訓練權重；沒有權重時，網頁仍可顯示特徵分析。

原本公開追蹤的姿態 CSV 已從目前檔案移除，Git 歷史保留；先前提交仍可含該檔。新資料、模型與分析產物預設排除於 Git。正式研究資料請在自己的受控環境處理，這個專案是研究分析展示。

---

## English

# Analysis of Walking Posture in COPD Patients

**Department of Computer Science and Information Engineering, Asia University | Academic Year 113 graduation project**

This project uses gait videos of COPD patients provided by **Taichung Veterans General Hospital**. It combines **YOLOv7 keypoint extraction, gait feature analysis, and a multilayer perceptron (MLP)** to investigate classification into four severity categories and uses SHAP to examine feature contributions for research on quantitative activity assessment.

The project poster describes **71 COPD patients and approximately 60,000 data points**. The study focuses on identifying useful walking-posture features and connecting statistical visualization, model evaluation, and a results interface.

[Research source comparison](docs/report_alignment.md) · [Executable notebook](notebooks/gait_workflow.ipynb) · [Data format](docs/data_format.md)

### Research workflow

1. **Video and keypoint extraction:** convert gait videos into individual frames, extract human keypoints with YOLOv7, and export CSV files.
2. **Preprocessing:** review incorrect detections and missing data, split KPTS coordinate fields, and standardize features with StandardScaler.
3. **Gait analysis:** examine head–torso angle, arm swing angle, step distance/length, hip angle, and statistics such as means and standard deviations; compare classes using quartiles, boxplots, and outlier analysis.
4. **Four-class classification:** train a PyTorch MLP for the four COPD patient categories used in the study.
5. **Interpretation and presentation:** evaluate confusion matrices, precision, recall, F1, ROC/PR curves, and SHAP, and present a gait assessment interface.

The architecture shown on report pages 31–33 has **three hidden layers of 32 → 64 → 16 units and a four-class output layer**, making four fully connected layers. It uses ReLU, dropout **0.1**, Adam, and learning rate **0.001**. The report text states 30 input features; see the [source comparison](docs/report_alignment.md) for the limitations in matching complete feature lists to individual experiments.

### Reported project results

| Item | Record in the source documents |
| --- | --- |
| Test accuracy | **79.07%** in the screenshot on report page 35; the text on that page describes approximately **72%–79%** |
| Precision | **0.76** |
| Recall | **0.79** |
| F1 | **0.77** |
| Poster summary | **80%** accuracy |

These figures are presented separately as recorded in the report and poster. The reported experiment was not rerun during this documentation update.

The report's SHAP analysis highlights **mean head–torso angle, the standard deviation of shoulder–hip distance, and the standard deviation of step length**. The more numerous first and second categories performed better, while the third and fourth categories were harder to classify. Class imbalance and dataset size were identified as priorities for further work, alongside expanded patient data, improved features and models, and the results interface.

### Repository contents

The repository contains preprocessing, gait statistics, MLP classification, and SHAP-related code. It includes **30 gait research notebooks selected from 127 Colab notebooks**, the GitHub training source, and an executable CSV analysis and Flask demonstration workflow. See the [source review](docs/source_inventory.md) and [complete index](docs/source_index.md).

The executable entry points currently start with **keypoint CSV files**. Video frame extraction and YOLOv7 belong to the study workflow but are not packaged here as a one-command video analysis feature. The research description follows the report and poster; the operating settings below describe the actual code in this repository.

### Repository layout

| Location | Purpose |
| --- | --- |
| `app.py`, `templates/`, `static/css/` | CSV upload and web results |
| `analyze.py` | Feature, statistics, and chart exports |
| `train.py` | MLP and Random Forest training with subject-level splitting |
| `explain.py` | Random Forest SHAP |
| `utils/data_processor.py` | Shared parsing and feature definitions |
| `models/analyzer.py`, `models/predict.py` | MLP architecture, model validation, inference |
| `examples/make_demo.py` | Synthetic data generator |
| `notebooks/gait_workflow.ipynb` | Runnable integrated notebook |
| `research/` | 30 Colab research notebooks and the original GitHub training source |
| `docs/`, `tests/` | Data format, source review, example figure, and tests |
| `requirements.txt`, `requirements-explain.txt` | Runtime and optional SHAP dependencies |

The research notebooks retain exploratory work with outputs and Colab metadata removed. They contain independent experiments and inconsistent coordinate definitions, so they are not guaranteed to run sequentially. Use the root-level programs or the integrated notebook for the executable workflow.

### Quick start

Python 3.11 or 3.12 is recommended. Run from the repository root; create and activate a virtual environment first if desired.

```bash
python -m pip install -r requirements.txt
python examples/make_demo.py --output data/demo
python analyze.py data/demo/synthetic_pose.csv --output results/demo
python app.py
```

![Synthetic workflow example](docs/images/synthetic_feature_traces.png)

This figure uses synthetic coordinates to demonstrate the software; it is not a figure from the patient experiment.

Open <http://127.0.0.1:5000> and select `data/demo/synthetic_pose.csv`. The analysis command exports per-row features, recording summaries, quartile JSON, boxplots, and line plots. Web uploads are processed in memory, with limits of 10 MB and 100000 rows; there is no route for downloading raw uploads.

The integrated notebook also runs synthetic data generation, analysis, a training smoke test, and SHAP. In Colab, it first obtains this repository and installs dependencies.

### Your own data

Each CSV must contain **one recording of one tracked person**, using a `kpts` field with 51 values or columns `kpt_1` through `kpt_51`. The executable workflow explicitly requires **COCO 17 keypoints in x/y/confidence order**, with x increasing rightward and y downward. Verify the actual source schema first and convert other layouts before use. See the [data format and formulas](docs/data_format.md).

Horizontal ankle separation is an image-coordinate x difference, not calibrated physical step length. The torso midpoint is not the body's center of mass. Without reliable timing or frame rate, row numbers are not used to infer cadence, velocity, or acceleration. IQR outliers remain in the data and are flagged; low-confidence points are not replaced with zeros.

For a CSV containing multiple tracking IDs, enter the desired ID in the web form or use `python analyze.py data/recording.csv --track-id 1`. Before training, separate people and recordings into individual files. A tracking ID is not a cross-video subject identity.

Prepare `data/manifest.csv` for classification. Relative recording paths resolve from the manifest's directory:

```csv
path,subject_id,label
recording_001.csv,subject_001,class_A
recording_002.csv,subject_002,class_B
```

These rows illustrate the format only. Training requires at least two distinct subjects per class and a test split large enough to contain every class. All recordings of a person must share the same subject ID and label. Labels come from your study manifest; the software does not interpret them as disease stages or normal/abnormal status.

### Running training and SHAP

```bash
python train.py data/manifest.csv --output artifacts/run --epochs 100 --test-size 0.25 --seed 42
python -m pip install -r requirements-explain.txt
python explain.py artifacts/run/random_forest.joblib artifacts/run/test_features.csv --output results/shap
```

The current `train.py` uses 22 mean/standard-deviation features per recording. Subjects are split with class stratification, and standardization is fitted on training data only. This executable entry point retains the GitHub web model with hidden layers of 256→128→64 and dropout 0.45, which differs from the report architecture described above. Random Forest uses 200 trees and maximum depth 8. `explain.py` explains the Random Forest; this does not attribute the reported MLP classification scores to Random Forest.

Training writes `model.pth`, `random_forest.joblib`, `metrics.json`, `split.csv`, and `test_features.csv`. The MLP artifact includes feature order, coordinate schema, standardization parameters, and labels. Place your trained `model.pth` at `models/model.pth` and restart the web app to enable inference. Load only model files you created or trust.

The default `--threshold 0` accepts every classification. A higher preselected threshold reports overall accuracy, accepted-recording accuracy, and coverage together. Tune thresholds and parameters using a separate validation set and reserve the test set for final evaluation. Metrics are recording-level; a subject may contribute multiple recordings. Actual studies must document sample sizes, class distribution, and splitting decisions.

### Verification and available material

```bash
python -m unittest discover -s tests -v
```

Tests cover known geometry, triplet mapping, low confidence, invalid inputs, disjoint subject splits, training/inference compatibility, threshold coverage, and web uploads. Run `python train.py data/demo/manifest.csv --output artifacts/demo --epochs 2` for a synthetic smoke test; its scores do not measure research performance.

The tested environment and completed checks are listed in the [verification record](docs/verification.md).

The research description and results have been checked against the supplied report and poster. Page references and code differences are recorded in the [source comparison](docs/report_alignment.md). The repository provides code, research notebooks, and synthetic examples, but not the complete labeled dataset and trained weights needed to reproduce the reported experiment. Feature analysis works without model weights.

The previously tracked pose CSV has been removed from the current tree while preserving Git history; earlier commits may still contain it. New data, models, and analysis artifacts are ignored by Git by default. Process study data in your own controlled environment. This project is a research analysis demonstration.
