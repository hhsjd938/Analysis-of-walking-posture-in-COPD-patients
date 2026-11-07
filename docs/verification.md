# 驗證記錄 / Verification

- 15 項自動化測試通過：已知幾何、座標解析、低信心值、單一追蹤對象選取、分組切分、模型保存與載入、閾值覆蓋率、網頁與 API。
- 合成資料完成 CSV → 特徵／圖表 → MLP 與 Random Forest 訓練 → SHAP 匯出。MLP 兩輪訓練只做煙霧測試，不代表研究效能。
- 32 份筆記本格式驗證通過：30 份 Colab 來源、1 份 GitHub 訓練來源及 1 份整合筆記本。研究來源筆記不保證逐格執行。
- README 與文件內的本地連結、示例 PNG 及 Python 語法通過檢查。
- GitHub 原始姿態 CSV 僅核對格式與追蹤 ID，未用於訓練或發布新研究結果。

15 automated tests passed. The synthetic workflow completed feature and chart export, MLP and Random Forest training, and SHAP export. All 32 notebooks passed format validation; exploratory source notebooks were not executed as sequential pipelines. Documentation links, the PNG and Python syntax were checked. The original pose CSV was inspected structurally, not used to train or publish study results.

## 測試環境 / Test environment

Python 3.12 / Windows. These are the versions used for validation; dependency ranges are defined in requirements files.

| Package | Version |
| --- | --- |
| Flask | 3.1.3 |
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| matplotlib | 3.11.2 |
| scikit-learn | 1.9.1 |
| torch | 2.15.0 |
| shap | 0.50.0 |
