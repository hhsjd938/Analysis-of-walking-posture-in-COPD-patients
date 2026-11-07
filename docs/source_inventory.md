# Colab 資料盤點

逐份讀取指定 Drive 資料夾的 127 份筆記本原始碼後，確認 30 份含步態專題的關鍵點拆解、角度、踝部座標、四分位數、IQR 或分類流程。另 97 份未納入，其中包含空白檔及其他主題。判定依據是實際程式及資料欄位，不只依檔名。

## 納入的 30 份

| Drive 原檔名 | 主題依據 | 整理後位置 |
| --- | --- | --- |
| COPD機器學習(進階修改架構).ipynb | 分類網路與評估 | [筆記](../research/classification/copd_network.ipynb) |
| calculate&步距.ipynb | 關鍵點拆解與踝部座標 | [筆記](../research/coordinates/ankle_separation.ipynb) |
| 學習測試(進階修改架構)3.ipynb | 分類網路與評估 | [筆記](../research/classification/network_experiment_3.ipynb) |
| 學習測試(進階修改架構)2.ipynb | 分類網路與評估 | [筆記](../research/classification/network_experiment_2.ipynb) |
| 學習測試(進階修改架構)(側面).ipynb | 分類網路與評估 | [筆記](../research/classification/side_view_network.ipynb) |
| 頭部與軀幹角度&盒需圖.ipynb | 角度計算與圖表 | [筆記](../research/angles/head_torso_boxplot.ipynb) |
| 手臂擺幅角度&盒需圖.ipynb | 角度計算與圖表 | [筆記](../research/angles/arm_boxplot.ipynb) |
| Q1Q2Q3角度盒需圖比對.ipynb | 四分位數比較 | [筆記](../research/quartiles/angle_comparison_a.ipynb) |
| Q1Q2Q3步距盒需圖比對.ipynb | 四分位數比較 | [筆記](../research/quartiles/ankle_comparison.ipynb) |
| 步頻標準差盒需圖.ipynb | 關鍵點拆解與踝部座標 | [筆記](../research/coordinates/motion_statistics.ipynb) |
| 學習測試(基礎架構).ipynb | 分類網路與評估 | [筆記](../research/classification/basic_network.ipynb) |
| 新步距盒需圖測試(消除離群值).ipynb | IQR 與離群檢查 | [筆記](../research/outliers/ankle_iqr.ipynb) |
| 新角度盒需圖測試(消除離群).ipynb | IQR 與離群檢查 | [筆記](../research/outliers/angle_iqr.ipynb) |
| 找離群.ipynb | IQR 與離群檢查 | [筆記](../research/outliers/locate_outliers.ipynb) |
| 步距Q1&圖.ipynb | 四分位數比較 | [筆記](../research/quartiles/ankle_q1.ipynb) |
| Q1角度&圖&極距.ipynb | 四分位數比較 | [筆記](../research/quartiles/angle_q1_range.ipynb) |
| Q1Q2Q3角度盒需圖比對(測試).ipynb | 四分位數比較 | [筆記](../research/quartiles/angle_comparison_experiment.ipynb) |
| Q1Q2Q3角度盒需圖比對.ipynb | 四分位數比較 | [筆記](../research/quartiles/angle_comparison_b.ipynb) |
| Q1Q2Q3角度折線圖比對.ipynb | 四分位數比較 | [筆記](../research/quartiles/angle_lines.ipynb) |
| Q3角度.ipynb | 四分位數比較 | [筆記](../research/quartiles/angle_q3.ipynb) |
| Q2角度.ipynb | 四分位數比較 | [筆記](../research/quartiles/angle_q2.ipynb) |
| 步距Q2&圖.ipynb | 四分位數比較 | [筆記](../research/quartiles/ankle_q2.ipynb) |
| 步距Q3&圖.ipynb | 四分位數比較 | [筆記](../research/quartiles/ankle_q3_a.ipynb) |
| 步距Q3&圖.ipynb | 四分位數比較 | [筆記](../research/quartiles/ankle_q3_b.ipynb) |
| calculate&angle_change.ipynb | 角度計算與圖表 | [筆記](../research/angles/angle_change.ipynb) |
| 步距75%計算.ipynb | 四分位數比較 | [筆記](../research/quartiles/ankle_percentile_75.ipynb) |
| Untitled29.ipynb | 角度計算與圖表 | [筆記](../research/angles/joint_vectors.ipynb) |
| 正面脊椎角度測試.ipynb | 角度計算與圖表 | [筆記](../research/angles/frontal_trunk.ipynb) |
| Untitled28.ipynb | 關鍵點拆解與踝部座標 | [筆記](../research/coordinates/ankle_positions.ipynb) |
| Untitled0.ipynb | 關鍵點拆解與踝部座標 | [筆記](../research/coordinates/ankle_coordinate_experiment.ipynb) |

未命名的 Untitled29、Untitled28、Untitled0 分別包含關節向量、踝部位置、踝部座標處理，因此納入。重複檔名加上 a/b 以保留各自內容；不表示版本先後。

## 未納入的內容

- `正面脊椎q1q2q3.ipynb` 是空白筆記本，沒有可整理的程式。
- `資料科學在教育期末.ipynb` 使用教育資料 edu_bigdata，末尾雖有一般角度函式，沒有步態資料流程，因此未納入。
- 教育成績預測、MLP／邏輯迴歸／CFT 課程實驗屬另一個作品集。
- 胸部 X 光、MedGemma、影像生成、心衰竭用藥、新生兒資料、交通、金融、MNIST、花卉分類及一般程式練習，不屬本專題。
- HMI、抽籤及其他空白筆記本也未納入。

## 原 GitHub 內容整合

`app.py`、HTML、樣式及 `models/analyzer.py` 保留為展示程式；資料處理統一到 `utils/data_processor.py`。原訓練程式收入 [GitHub 訓練筆記](../research/github_training.ipynb)，整合入口為 `train.py`。移除空白模型、空白 JavaScript、編譯快取；補上依賴、合成示例及測試。

原先追蹤的姿態 CSV 已從目前公開檔案移除，整理時留存本機副本；Git 歷史保留，因此先前提交仍含該 CSV。公開示例全部由程式合成。筆記本只保留程式與文字，不發布執行輸出或 Colab 使用者資訊。

該 CSV 共 1290 列、4 個追蹤 ID。整合分析會拒絕直接混合多個 ID，可在網頁或命令列選定其中一個；訓練資料仍需先依對象與記錄分檔。

## 研究筆記與整合流程的差異

1. 來源以 kpt_1…kpt_51 表示 17 組三元數值，部分程式卻連續取前 18 欄當 x/y。整合流程明訂 COCO 17 點的 x/y/confidence；若來源模型配置不同，需先轉換，不能直接套用。
2. 來源有將髖部寬度命名為步距，或以座標差除以列號當速度的實驗。整合流程改用「踝部水平間距」，不聲稱是實際步長、步頻、速度或加速度；軀幹中點也不稱為人體質心。
3. 四分位數是各訊號有效列的 Q1/Q2/Q3；1.5×IQR 只標記離群值，不自動刪除。低信心點與零長度向量不填成 0 度；完全無有效值時回報錯誤。
4. 分類以記錄為單位建立 22 個特徵，以 subject_id 分割，標準化只用訓練資料。保留 GitHub 的 256→128→64 MLP 與來源中的 Random Forest／SHAP 方法；探索筆記中的其他網路架構仍可查閱。
5. 筆記包含手填混淆矩陣、隨機示例、重用變數及測試集閾值掃描。這些不搬成研究成果；整合評估同時報告全體準確率、覆蓋率及接受樣本準確率。閾值需事先設定或用額外驗證集決定。

本次未在提供的資料夾找到畢業專題報告 PDF／Word，也沒有可用的訓練權重與受試者標籤表；未建立或宣稱新的臨床研究結果。
