# 驗證記錄 / Verification

- 15 項自動化測試通過：已知幾何、座標解析、低信心值、單一追蹤對象選取、分組切分、模型保存與載入、閾值覆蓋率、網頁與 API。
- 合成資料完成 CSV → 特徵／圖表 → MLP 與 Random Forest 訓練 → SHAP 匯出。MLP 兩輪訓練只做煙霧測試，不代表研究效能。
- README 與文件內的本地連結、示例 PNG 及 Python 語法通過檢查。

15 automated tests passed. The synthetic workflow completed feature and chart export, MLP and Random Forest training, and SHAP export. Documentation links, the PNG and Python syntax were checked.

MLP 訓練與網頁推論共用 32→64→16、Dropout 0.1 的定義。

Training and web inference share the 32→64→16 MLP definition with dropout 0.1.

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
