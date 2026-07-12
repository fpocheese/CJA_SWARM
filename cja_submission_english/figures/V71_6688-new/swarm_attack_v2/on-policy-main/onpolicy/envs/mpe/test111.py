import numpy as np

G = 9.81  # 重力加速度，单位 m/s^2

class Missile:
    def __init__(self, position, velocity, target_position, navigation_constant=3):
        """
        初始化导弹参数
        :param position: 初始位置 [x, y]
        :param velocity: 初始速度 [vx, vy]
        :param target_position: 目标位置 [tx, ty]
        :param navigation_constant: 比例导引常数
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.target_position = np.array(target_position, dtype=float)
        self.navigation_constant = navigation_constant
        self.axial_overload = 0  # 轴向过载
        self.normal_overload = 0  # 法向过载

    def compute_time_to_target(self):
        """
        计算当前导弹到达目标的预计时间
        """
        relative_position = self.target_position - self.position
        distance = np.linalg.norm(relative_position)
        speed = np.linalg.norm(self.velocity)
        return distance / max(speed, 1e-3)  # 避免除以零

    def update_guidance(self, dt, max_time_to_target):
        """
        更新导弹的轴向和法向过载
        :param dt: 时间步长
        :param max_time_to_target: 当前所有导弹中最大预计到达时间
        """
        # 计算相对位置和速度
        relative_position = self.target_position - self.position
        distance = np.linalg.norm(relative_position)
        unit_vector_to_target = relative_position / distance

        # 当前速度分解为轴向和法向分量
        axial_speed = np.dot(self.velocity, unit_vector_to_target)

        # 动态调整轴向过载
        required_speed = distance / max_time_to_target
        self.axial_overload = (required_speed - axial_speed) / dt
        self.axial_overload = np.clip(self.axial_overload, -0.01 * G, 1 * G)

        # 法向过载：使用比例导引
        relative_velocity = self.velocity - [0, 0]  # 假设目标静止
        line_of_sight_rate = np.cross(relative_position, relative_velocity) / (distance**2)
        self.normal_overload = self.navigation_constant * np.linalg.norm(self.velocity) * line_of_sight_rate
        self.normal_overload = np.clip(self.normal_overload, -1 * G, 1 * G)

        # 更新速度
        axial_acceleration = self.axial_overload * unit_vector_to_target
        normal_acceleration = self.normal_overload * np.array([-unit_vector_to_target[1], unit_vector_to_target[0]])
        self.velocity += (axial_acceleration + normal_acceleration) * dt

        # 限制最大速度
        speed = np.linalg.norm(self.velocity)
        if speed > 40:
            self.velocity = self.velocity / speed * 40

        # 更新位置
        self.position += self.velocity * dt

    def is_target_reached(self):
        """判断是否到达目标"""
        return np.linalg.norm(self.target_position - self.position) < 1e-2


# 初始化导弹参数
missiles = [
    Missile(position=[0, 0], velocity=[25, 0], target_position=[2000, 0]),
    Missile(position=[0, 1000], velocity=[20, 0], target_position=[2000, 0]),
    Missile(position=[0, -1000], velocity=[30, 0], target_position=[2000, 0])
]

dt = 0.1  # 时间步长
time = 0
while any(not missile.is_target_reached() for missile in missiles):
    # 计算所有导弹的预计到达时间
    times_to_target = [missile.compute_time_to_target() for missile in missiles if not missile.is_target_reached()]
    max_time_to_target = max(times_to_target)  # 取最大预计到达时间

    print(f"Time: {time:.1f}s")
    for i, missile in enumerate(missiles):
        if not missile.is_target_reached():
            missile.update_guidance(dt, max_time_to_target)
            print(f"  Missile {i+1}: Position={missile.position}, Velocity={missile.velocity}, "
                  f"Axial Overload={missile.axial_overload:.2f}, Normal Overload={missile.normal_overload:.2f}")
    time += dt

print("All missiles reached the target.")
