import pandas as pd
import numpy as np
import math

def calculate_head_torso_angle(x_head, y_head, x_shoulder, y_shoulder):
    delta_x = x_head - x_shoulder
    delta_y = y_head - y_shoulder
    angle_radians = math.atan2(delta_y, delta_x)
    return math.degrees(angle_radians)

def calculate_arm_angle(x_shoulder, y_shoulder, x_elbow, y_elbow, x_wrist, y_wrist):
    a_x = x_elbow - x_shoulder
    a_y = y_elbow - y_shoulder
    b_x = x_wrist - x_elbow
    b_y = y_wrist - y_elbow
    dot_product = a_x * b_x + a_y * b_y
    magnitude_a = math.hypot(a_x, a_y)
    magnitude_b = math.hypot(b_x, b_y)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    cos_angle = dot_product / (magnitude_a * magnitude_b)
    cos_angle = max(-1, min(1, cos_angle))
    return math.degrees(math.acos(cos_angle))

def calculate_hip_angle(x_shoulder, y_shoulder, x_hip, y_hip, x_knee, y_knee):
    a_x = x_shoulder - x_hip
    a_y = y_shoulder - y_hip
    b_x = x_knee - x_hip
    b_y = y_knee - y_hip
    dot_product = a_x * b_x + a_y * b_y
    magnitude_a = math.hypot(a_x, a_y)
    magnitude_b = math.hypot(b_x, b_y)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    cos_angle = dot_product / (magnitude_a * magnitude_b)
    cos_angle = max(-1, min(1, cos_angle))
    return math.degrees(math.acos(cos_angle))

def extract_features_from_rows(df, file_name, label):
    head_torso_angles = []
    arm_angles = []
    hip_angles = []
    row_features = []

    for i in range(len(df)):
        try:
            # 提取數據點
            x_head = (float(df['kpt_1'].iloc[i]) + float(df['kpt_2'].iloc[i])) / 2
            y_head = (float(df['kpt_3'].iloc[i]) + float(df['kpt_4'].iloc[i])) / 2
            x_shoulder = (float(df['kpt_5'].iloc[i]) + float(df['kpt_6'].iloc[i])) / 2
            y_shoulder = (float(df['kpt_7'].iloc[i]) + float(df['kpt_8'].iloc[i])) / 2
            x_elbow = float(df['kpt_9'].iloc[i])
            y_elbow = float(df['kpt_10'].iloc[i])
            x_wrist = float(df['kpt_11'].iloc[i])
            y_wrist = float(df['kpt_12'].iloc[i])
            x_hip_left = float(df['kpt_13'].iloc[i])
            y_hip_left = float(df['kpt_14'].iloc[i])
            x_hip_right = float(df['kpt_15'].iloc[i])
            y_hip_right = float(df['kpt_16'].iloc[i])
            x_knee = float(df['kpt_17'].iloc[i])
            y_knee = float(df['kpt_18'].iloc[i])

            # 計算特徵
            head_torso_angle = calculate_head_torso_angle(x_head, y_head, x_shoulder, y_shoulder)
            arm_angle = calculate_arm_angle(x_shoulder, y_shoulder, x_elbow, y_elbow, x_wrist, y_wrist)
            hip_angle = calculate_hip_angle(
                x_shoulder, y_shoulder,
                (x_hip_left + x_hip_right) / 2, (y_hip_left + y_hip_right) / 2,
                x_knee, y_knee
            )

            row_features.append({
                'head_torso_angle': head_torso_angle,
                'arm_angle': arm_angle,
                'hip_angle': hip_angle,
                'label': label,
                'file_name': file_name
            })

            head_torso_angles.append(head_torso_angle)
            arm_angles.append(arm_angle)
            hip_angles.append(hip_angle)

        except Exception as e:
            print(f"特徵提取錯誤於文件 {file_name}, 行 {i}: {e}")
            continue

    statistical_features = {
        'mean_head_torso_angle': np.mean(head_torso_angles),
        'std_head_torso_angle': np.std(head_torso_angles),
        'mean_arm_angle': np.mean(arm_angles),
        'std_arm_angle': np.std(arm_angles),
        'mean_hip_angle': np.mean(hip_angles),
        'std_hip_angle': np.std(hip_angles),
        'label': label,
        'file_name': file_name
    }

    return {'row_features': row_features, 'statistical_features': statistical_features}