# 資料格式與特徵定義 / Data format and feature definitions

本文定義目前 CSV 分析程式的輸入與計算方式。報告中的 YOLOv7、特徵名稱及當時公式請對照[研究資料對照](report_alignment.md)；這些定義不代表逐項重現當年的實驗。

This document specifies the current CSV analysis code. See the [research comparison](report_alignment.md) for YOLOv7 and the feature names/formulas in the report; these definitions do not establish an exact reproduction of that experiment.

## 輸入 / Input

單一 CSV = 單一錄製片段、單一追蹤對象。若有 `id` 欄，僅允許一個值；沒有 `id` 時，呼叫端負責先分離對象。列順序由輸入保留，不排序、不當成時間戳。不同拍攝方向與座標尺度不可直接混用而不記錄條件。

One CSV represents one recording of one person. An `id` column, when present, must have one unique value. Otherwise the caller must isolate the person first. Input row order is preserved, not sorted or treated as timestamps. Record viewing direction and coordinate scale when comparing recordings.

支援 `kpts` 欄的逗號分隔數值／JSON 陣列，或 51 個完整展開欄位 `kpt_1`…`kpt_51`。每個點為 x、y、confidence，confidence 範圍為 [0,1]。不接受 NaN／Infinity、缺少欄位或錯誤長度。兩種格式同時存在時，以完整展開欄位為準。

Supported formats are a comma-separated or JSON-array `kpts` field, or all 51 expanded `kpt_1`…`kpt_51` columns. Each point is x, y, confidence, with confidence in [0,1]. Nonfinite values, missing columns, and incorrect lengths are rejected. Complete expanded columns take precedence if both formats are present.

## COCO 17 點配置 / Keypoint layout

以下採用 [Ultralytics 官方姿態文件中的 COCO 17 點順序](https://docs.ultralytics.com/tasks/pose/)。這是整合程式的明確輸入契約，不代表已確認所有來源影片的姿態模型。來源中 kpt_46／49 作為兩踝 x 的用法與此配置相符，其他來源部分角度索引不一致，因此不可直接沿用來源角度數值比較。

The integrated program uses the [COCO 17-point ordering documented by Ultralytics](https://docs.ultralytics.com/tasks/pose/). This is an explicit input contract, not confirmation of the pose model used for every source video. The source use of kpt_46/49 as ankle x coordinates is consistent with it, while some source angle indexing is not. Do not directly compare the resulting angles with those exploratory outputs.

| 零起算 index | 點 / Point | x, y, confidence 欄位 |
| --- | --- | --- |
| 0 | 鼻 / Nose | 1, 2, 3 |
| 1, 2 | 左右眼 / Eyes | 4–9 |
| 3, 4 | 左右耳 / Ears | 10–15 |
| 5, 6 | 左右肩 / Shoulders | 16–21 |
| 7, 8 | 左右肘 / Elbows | 22–27 |
| 9, 10 | 左右腕 / Wrists | 28–33 |
| 11, 12 | 左右髖 / Hips | 34–39 |
| 13, 14 | 左右膝 / Knees | 40–45 |
| 15, 16 | 左右踝 / Ankles | 46–51 |

## 逐列特徵 / Per-row features

S 是左右肩中點，H 是左右髖中點，N 是鼻。∠ABC 為以 B 為頂點的內角，使用點積計算、範圍 0–180 度；零長度向量為缺值。座標 x 向右、y 向下。

S is the shoulder midpoint, H is the hip midpoint, and N is the nose. ∠ABC is the interior angle at B, computed with the dot product in degrees from 0 to 180. Zero-length vectors produce missing values. x increases rightward and y downward.

| 特徵 / Feature | 定義 / Definition |
| --- | --- |
| head_torso_angle | ∠NSH |
| left/right_elbow_angle | 同側肩–肘–腕的內角 / Ipsilateral shoulder–elbow–wrist angle |
| left/right_hip_angle | 同側肩–髖–膝的內角 / Ipsilateral shoulder–hip–knee angle |
| trunk_tilt | atan2(Sx−Hx, −(Sy−Hy))，度 / degrees；直立為 0 / upright is 0 |
| ankle_horizontal_separation | abs(left_ankle_x − right_ankle_x) |
| shoulder_hip_distance | Euclidean distance(S, H) |
| head_hip_distance | Euclidean distance(N, H) |
| torso_midpoint_x/y | (S + H) / 2 |

預設 confidence < 0.25 的點無效；只有依賴這個點的特徵受影響。每個訊號分別使用有效列計算平均、母體標準差 ddof=0、四分位數、極距及離群數。若任一訊號全無有效值則停止，不用 0 補齊。IQR=Q3−Q1，以小於 Q1−1.5IQR 或大於 Q3+1.5IQR 標記；不刪除資料、不以測試資料擬合過濾器。

Points below confidence 0.25 are missing by default; only dependent features are affected. Each signal uses its valid rows for mean, population standard deviation (ddof=0), quartiles, range, and outlier count. Processing stops if a signal has no valid values. Values below Q1−1.5IQR or above Q3+1.5IQR are flagged, with IQR=Q3−Q1. Rows are not removed and no filter is fitted using test data.

距離與中點沿用原始座標單位；不推定公分／公尺。22 個模型特徵固定為上述 11 個訊號各自的 mean/std，順序由 `FEATURE_COLUMNS` 決定，不含檔名、ID、標籤。拍攝方向、視角、影像大小及有效列比例都可能影響比較；目前未做相機校正、步態週期切分或跨視角校正。trunk_tilt 接近 ±180 度時有角度環繞問題，本流程使用一般算術統計，需在實驗中檢查資料分布。

Distances and midpoints retain the original coordinate units; no centimeter/meter scale is assumed. The 22 model features are the mean/std of the 11 signals in `FEATURE_COLUMNS` order, excluding filenames, IDs, and labels. View direction, camera angle, image size, and valid-row proportions affect comparability. Camera calibration, gait-cycle segmentation, and cross-view correction are not implemented. Trunk tilt wraps at ±180 degrees; this workflow uses arithmetic statistics, so inspect angle distributions in actual experiments.
