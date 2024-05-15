import numpy as np


class KalmanFilter2D:
    def __init__(self, initial_state, process_variance=1e-5, measurement_variance=1e-4, ):

        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = initial_state  # 后验估计值（卡尔曼滤波器的输出）
        self.posteri_error_estimate = 1.0  # 后验估计误差的方差

    def update(self, measurement):
        # 预测更新
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # 测量更新
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate


if __name__ == '__main__':
    process_variance = 1e-5  # 过程噪声的方差
    measurement_variance = 1e-1  # 观测噪声的方差
    initial_state = np.array([0, 0])  # 初始状态估计值
    kf = KalmanFilter2D(initial_state, process_variance, measurement_variance, )
