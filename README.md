# COPD 患者行走姿態分析

**畢業專題｜Analysis of Walking Posture in COPD Patients**

以人體姿態關鍵點分析行走時的關節角度與座標變化，整理從 CSV 資料處理、四分位數與離群檢查，到分類模型、特徵解釋及網頁展示的研究流程。

本作品集整合 GitHub 原有程式，以及從 127 份 Colab 筆記本中確認與步態專題相關的 30 份研究筆記。完整判定與來源對應見[資料盤點](docs/source_inventory.md)及[全部來源索引](docs/source_index.md)。

[English](#english) · [可執行筆記本](notebooks/gait_workflow.ipynb) · [資料格式](docs/data_format.md) · [驗證記錄](docs/verification.md)

## 專題內容

- **座標處理：**解析 17 組關鍵點，檢查格式與信心分數，避免把 confidence 當成 x/y。
- **步態特徵：**頭部與軀幹角度、左右肘角、左右髖角、軀幹傾斜、踝部水平間距及軀幹座標摘要。
- **統計與視覺化：**逐列訊號、平均、標準差、Q1/Q2/Q3、極距、1.5×IQR 離群標記、盒鬚圖與折線圖。
- **分類與解釋：**PyTorch MLP、Random Forest，以及 Random Forest 的 SHAP 特徵貢獻。
- **網頁展示：**Flask 上傳 CSV、顯示特徵摘要與四分位數；提供相容權重後可顯示分類結果。

![合成座標特徵示例](docs/images/synthetic_feature_traces.png)

上圖由程式產生的合成座標繪製，用來展示分析流程，不是病患研究結果。

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

## 訓練與 SHAP

```bash
python train.py data/manifest.csv --output artifacts/run --epochs 100 --test-size 0.25 --seed 42
python -m pip install -r requirements-explain.txt
python explain.py artifacts/run/random_forest.joblib artifacts/run/test_features.csv --output results/shap
```

模型使用每段記錄的 22 個平均／標準差特徵；以受試者分組並按類別切分，標準化只在訓練集擬合。MLP 保留原 GitHub 的 256→128→64 隱藏層與 Dropout 0.45；Random Forest 使用 200 棵樹、最大深度 8。SHAP 只解釋 Random Forest，不代表 MLP 或因果關係。

訓練產生 `model.pth`、`random_forest.joblib`、`metrics.json`、`split.csv` 及 `test_features.csv`。MLP 權重同時保存特徵順序、座標格式、標準化參數與類別名稱。將自己訓練的 `model.pth` 放到 `models/model.pth`，重新啟動網頁即可使用；只載入自己產生或可信任的模型檔。

預設 `--threshold 0` 不拒絕分類。若事先設定較高門檻，評估會同時列出全體準確率、接受樣本準確率與覆蓋率，不隱藏未分類樣本。門檻或參數調整須使用額外驗證集，保留測試集作最終評估。指標以「記錄」計算，同一受試者可能有多段記錄；資料量、類別分布與分割方式仍須在實際研究中交代。

## 驗證與資料狀態

```bash
python -m unittest discover -s tests -v
```

測試涵蓋已知幾何角度、三元座標映射、低信心值、錯誤輸入、受試者不跨集合、訓練／推論格式、閾值覆蓋率及網頁上傳。合成資料可用 `python train.py data/demo/manifest.csv --output artifacts/demo --epochs 2` 做流程煙霧測試，數值不代表研究效能。

目前提供程式、整理後筆記及合成示例。原儲存庫的權重檔是空白，提供的資料夾也沒有畢業專題報告 PDF／Word、可用權重或受試者標籤表，因此不填寫未驗證的準確率或臨床結論。沒有權重時，網頁仍可顯示特徵分析。

原本公開追蹤的姿態 CSV 已從目前檔案移除，Git 歷史保留；先前提交仍可含該檔。新資料、模型與分析產物預設排除於 Git。正式研究資料請在自己的受控環境處理，這個專案是研究分析展示。

---

## English

# Analysis of Walking Posture in COPD Patients

**Graduation project**

This portfolio analyzes joint angles and coordinate changes from human pose keypoints. It connects CSV processing, quartile and outlier analysis, classification, feature explanations, and a web demonstration.

It integrates the existing GitHub code with **30 gait-related notebooks identified from 127 Colab notebooks**. See the [source review](docs/source_inventory.md) and [complete source index](docs/source_index.md) for inclusion decisions and file mappings.

### Scope

- **Coordinate processing:** parse 17 keypoint triplets, validate confidence scores, and keep confidence separate from x/y coordinates.
- **Gait features:** head–torso angle, bilateral elbow and hip angles, trunk tilt, horizontal ankle separation, and torso coordinate summaries.
- **Statistics and visualization:** per-row signals, means, standard deviations, Q1/Q2/Q3, ranges, 1.5×IQR flags, boxplots, and line plots.
- **Classification and explanations:** a PyTorch MLP, Random Forest, and SHAP contributions for the Random Forest.
- **Web demonstration:** upload a CSV and inspect feature summaries and quartiles; classification becomes available with a compatible trained model.

![Synthetic coordinate feature example](docs/images/synthetic_feature_traces.png)

The figure uses programmatically generated synthetic coordinates. It demonstrates the workflow and is not a patient study result.

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

### Training and SHAP

```bash
python train.py data/manifest.csv --output artifacts/run --epochs 100 --test-size 0.25 --seed 42
python -m pip install -r requirements-explain.txt
python explain.py artifacts/run/random_forest.joblib artifacts/run/test_features.csv --output results/shap
```

Each recording supplies 22 mean/standard-deviation features. Subjects are split with class stratification, and standardization is fitted on training data only. The MLP retains the GitHub architecture with hidden layers of 256→128→64 and dropout 0.45. Random Forest uses 200 trees and maximum depth 8. SHAP explains the Random Forest, not the MLP or causal effects.

Training writes `model.pth`, `random_forest.joblib`, `metrics.json`, `split.csv`, and `test_features.csv`. The MLP artifact includes feature order, coordinate schema, standardization parameters, and labels. Place your trained `model.pth` at `models/model.pth` and restart the web app to enable inference. Load only model files you created or trust.

The default `--threshold 0` accepts every classification. A higher preselected threshold reports overall accuracy, accepted-recording accuracy, and coverage together. Tune thresholds and parameters using a separate validation set and reserve the test set for final evaluation. Metrics are recording-level; a subject may contribute multiple recordings. Actual studies must document sample sizes, class distribution, and splitting decisions.

### Verification and available material

```bash
python -m unittest discover -s tests -v
```

Tests cover known geometry, triplet mapping, low confidence, invalid inputs, disjoint subject splits, training/inference compatibility, threshold coverage, and web uploads. Run `python train.py data/demo/manifest.csv --output artifacts/demo --epochs 2` for a synthetic smoke test; its scores do not measure research performance.

The tested environment and completed checks are listed in the [verification record](docs/verification.md).

This repository provides code, curated notebooks, and synthetic examples. The original weight file was empty, and the supplied folder contained no graduation report PDF/Word document, usable weights, or subject-label manifest. No unverified accuracy or clinical conclusions are supplied. Feature analysis works without model weights.

The previously tracked pose CSV has been removed from the current tree while preserving Git history; earlier commits may still contain it. New data, models, and analysis artifacts are ignored by Git by default. Process study data in your own controlled environment. This project is a research analysis demonstration.
