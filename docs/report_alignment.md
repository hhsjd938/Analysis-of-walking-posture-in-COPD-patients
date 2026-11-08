# 專題報告與程式對照 / Research document comparison

2026-11-08 依使用者提供的兩份文件核對：

- `專 題 報 告 書.pdf`，41 頁，亞洲大學資訊工程學系 113 學年度專題報告，封面日期為民國 113 年 10 月。
- `步行姿態於_COPD_病患分析.pdf`，1 頁，專題成果海報。

以下頁碼指 PDF 頁面。文件用於核對研究敘述；原 PDF 未隨此次文字修正公開上傳。README 的研究摘要以這兩份文件為依據，執行方式以目前程式為依據。

## 研究內容與依據

| 項目 | 文件依據 | README 呈現 |
| --- | --- | --- |
| 研究目的 | 報告第 2、10–11 頁 | 使用步態特徵與 MLP 探索 COPD 四類病情分類及活動能力量化評估 |
| 資料來源 | 報告第 12–14 頁；海報實作方法 | 臺中榮民總醫院提供的病患步態影像 |
| 資料規模 | 海報實作方法 | 71 名 COPD 病患、約 60,000 筆資料點；資料點不等於獨立病患數 |
| 前處理 | 報告第 14–16 頁；海報實作方法 | 影片轉幀、YOLOv7、CSV、KPTS 拆解與標準化 |
| 特徵 | 報告第 16–30 頁 | 步距／步長、手臂擺幅、頭部與軀幹角度、髖關節角度及統計量 |
| MLP 架構 | 報告第 31 頁架構圖、第 32 頁程式截圖、第 33 頁設定 | 隱藏層 32→64→16、輸出 4；ReLU、Dropout 0.1、Adam、學習率 0.001 |
| 分類結果 | 報告第 35 頁截圖與文字 | 79.07%；Precision 0.76、Recall 0.79、F1 0.77；同頁文字描述約 72%–79% |
| 海報結果 | 海報實作方法 | 80%，單獨標為海報摘要，不替換報告的 79.07% |
| 特徵貢獻 | 報告第 37 頁 SHAP 圖、第 40 頁結論 | 平均頭部與軀幹角度、肩膀至髖部距離標準差、步長標準差 |
| 類別差異 | 報告第 38–40 頁 | 第一、第二類較好；第三、第四類受資料量與類別不平衡限制 |
| 結果展示 | 海報圖 2；報告第 41 頁 | 海報已有介面展示；報告將網頁整合列為後續方向，不據此推定已部署的臨床系統 |

四類是此研究資料中由輕至重的病患標籤；程式 Class 0–3 對應第一至第四類。文件沒有足夠的標籤定義可直接將其等同於特定臨床分期標準，也不能把第一類改稱健康對照組。

## 原文件之間需保留的差異

### 網路層數與特徵數

第 33 頁文字寫「隱藏層：4 層」，但列出的寬度只有 32、64、16；第 31–32 頁的圖與程式則清楚顯示三個隱藏層加一個輸出層。因此 README 使用「四層全連接，包含三個隱藏層及輸出層」。

第 33 頁記載 30 個輸入特徵，但研究筆記各實驗的統計欄位不同，輸入維度也由資料決定。尚未建立當年每次結果與完整特徵清單的一對一對應，因此保留「報告記載 30 個」，不把目前操作程式的 22 個特徵當成報告設定。

### 分割與評估

第 34 頁文字寫 70% 測試、30% 訓練，但同頁程式截圖是 `test_size=0.3`，對應 Colab 的 `split_data_by_file` 也以 30% 為測試比例。這支持程式採用訓練 70%／測試 30%，但文件文字有反向敘述，故將差異保留在此，不以未確認的單一比例概括所有結果。

海報提到交叉驗證，但未記載折數或每折結果；對應的主要訓練來源可見單次檔案分割。README 因此不將 80% 宣稱為已由現有程式重現的交叉驗證平均值。

第 38 頁混淆矩陣共有 43 個評估項目，對角線正確數為 34，34/43 約為 79.07%，與第 35 頁截圖一致。43 是該圖的評估項目數，不能直接解讀為 43 名獨立病患；原始標籤與分割清單仍需對照。

### 座標與特徵定義

報告第 14 頁稱每點為 x/y/z 且只使用 x/y；目前分析工具明訂第三值為 confidence。報告這段文字不足以確認第三值是實際三維深度，使用資料前仍須核對輸出格式。

報告的「步距」在第 16 頁描述為髖部水平距離，第 19 頁則使用兩個關鍵點的歐幾里得距離並以腳踝為例；研究筆記也有踝部水平差。這些不是相同量。README 在研究摘要保留當時的特徵名稱，操作文件則明確定義目前程式使用的踝部水平間距。

報告第 25–26 頁的頭部與軀幹角度使用頭部至肩部向量的 `atan2`；目前分析工具同名欄位使用鼻、肩中點、髖中點的內角，兩者不能直接對照角度數值。原始公式可在研究筆記查閱，操作公式見 [data_format.md](data_format.md)。

### SHAP 的解釋對象

報告與海報呈現 SHAP 圖，但沒有在圖上明確指明該次解釋器的模型物件。留存的主要訓練來源以 Random Forest 配合 FastTreeSHAP 計算貢獻，因此 README 不寫成「對 MLP 直接執行 TreeSHAP」。目前 `explain.py` 明確解釋 Random Forest，MLP 的報告成績也不因此歸為 Random Forest 成績。

## 程式入口的範圍

| 項目 | 報告／研究來源 | 目前可執行入口 |
| --- | --- | --- |
| 起始輸入 | 影片逐幀處理與 YOLOv7 關鍵點 | 已提取的關鍵點 CSV |
| 特徵 | 報告寫 30 個輸入；研究筆記有多種特徵集合 | 11 個訊號的 mean/std，共 22 個 |
| MLP | 32→64→16，Dropout 0.1 | GitHub 網頁模型 256→128→64，Dropout 0.45 |
| 分割 | 主要來源使用檔名分割，test_size=0.3 | subject_id 分組，預設測試比例 0.25 |
| 評估 | 報告截圖與海報成果記錄 | 合成資料流程測試；自備研究資料後另行訓練評估 |
| 權重 | 尚無可對應報告實驗的權重檔 | `train.py` 產生含特徵順序及標準化設定的權重 |

本次修正研究敘述及文件對應，沒有更改模型程式、重建病患資料或宣稱重現當年結果。原始程式與探索筆記仍可由 [來源盤點](source_inventory.md) 尋找。

## English

The README was checked on 2026-11-08 against a 41-page graduation report and a one-page project poster supplied by the author. The report is from Asia University's Department of Computer Science and Information Engineering, academic year 113, with an October 2024 cover date. The PDFs were used as references and were not published with this text update.

The report documents the research objective and hospital source on pages 2 and 10–14, preprocessing on pages 14–16, gait features on pages 16–30, and the MLP on pages 31–34. The poster supplies the cohort size of 71 patients and approximately 60,000 data points. These are different counting units.

The report screenshot on page 35 shows 79.07% accuracy, precision 0.76, recall 0.79, and F1 0.77. Its text gives an approximate range of 72%–79%; the poster separately states 80%. Page 38's confusion matrix contains 34 correct predictions out of 43 evaluated items, consistent with 79.07%, but does not establish that all 43 items are distinct patients. Pages 37 and 40 highlight mean head–torso angle, shoulder–hip distance standard deviation, and step-length standard deviation. Pages 38–40 describe better performance for the more numerous first and second categories. These categories should not be relabeled as a healthy control group or assumed to match a particular clinical staging standard.

The source comparison preserves the following distinctions:

- Pages 31–32 show three hidden layers (32, 64, 16) and a four-class output layer: four fully connected layers in total. Page 33's phrase “four hidden layers” conflicts with that diagram and code. Dropout is 0.1 and the learning rate is 0.001 with Adam.
- Page 33 states 30 inputs, while notebooks use varying feature sets and determine input size from data. The exact feature list for each reported run has not been established.
- Page 34's prose reverses the split relative to its `test_size=0.3` screenshot and matching notebook code. The code supports 70% training and 30% testing. The poster mentions cross-validation without folds or fold-level results; the main retained training source shows a single file-based split. The 80% figure is not presented as a reproduced cross-validation mean.
- The report calls keypoint triplets x/y/z, while the current CSV tool requires x/y/confidence. Its prose alone does not establish physical 3D depth. “Step distance” also varies between hip separation, ankle distance, and horizontal separation across sources. The report's head–torso angle uses `atan2` of a head-to-shoulder vector, whereas the current tool uses a nose–shoulder-midpoint–hip-midpoint interior angle. These quantities must not be treated as interchangeable.
- The report and poster show SHAP figures without clearly identifying the explained model object for that run. The retained main source uses Random Forest with FastTreeSHAP. The current `explain.py` explains Random Forest; reported MLP scores are not reassigned to that model.
- The current executable workflow starts from CSV, uses 22 features, retains the web model's 256→128→64 architecture with dropout 0.45, and splits by subject ID with a default test fraction of 0.25. Those settings describe the repository tools, not an exact reproduction of the reported experiment.
- The poster contains an assessment interface, while page 41 of the report lists web integration as future work. Neither establishes a deployed clinical system.

This update corrects documentation only. It does not change model code, reconstruct patient data, or claim to rerun the reported experiment. The [source inventory](source_inventory.md) links to the retained research notebooks.
