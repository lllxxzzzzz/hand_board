import numpy as np
import cv2


class KalmanFilter2D:
    def __init__(self, process_variance, measurement_variance, initial_state):
        """
        初始化2D卡尔曼滤波器。
        参数：
        - process_variance: 过程噪声的方差，用于模拟状态转移的不确定性。
        - measurement_variance: 观测噪声的方差，用于模拟传感器测量的不确定性。
        - initial_state: 初始状态的估计值，包括位置和速度等信息。
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = initial_state  # 后验估计值（卡尔曼滤波器的输出）
        self.posteri_error_estimate = 1.0  # 后验估计误差的方差

    def update(self, measurement):
        """
        更新卡尔曼滤波器的状态。

        参数：
        - measurement: 观测值，通常是从传感器获得的关键点位置。

        返回值：
        返回更新后的状态估计值。
        """
        # 预测更新
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # 测量更新
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate


# 模拟关键点检测（用你的实际检测代码替换此部分）
def detect_keypoint(frame):
    # 在固定位置（100, 100）模拟一个关键点
    return np.array([100, 100])


# 初始化卡尔曼滤波器
process_variance = 1e-5  # 过程噪声的方差
measurement_variance = 1e-1  # 观测噪声的方差
initial_state = np.array([0, 0])  # 初始状态估计值
kf = KalmanFilter2D(process_variance, measurement_variance, initial_state)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # 模拟关键点检测（用你的实际检测代码替换此部分）
    keypoint = detect_keypoint(frame)

    # 卡尔曼滤波
    filtered_keypoint = kf.update(keypoint.astype(float))

    # 在帧上绘制经过滤波的关键点
    cv2.circle(frame, tuple(filtered_keypoint.astype(int)), 5, (0, 255, 0), -1)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
