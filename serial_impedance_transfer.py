"""
串联阻抗传递模块
================

基于《Control of Humanoid Robots with Parallel Mechanisms.pdf》第二节D的理论实现
"串联空间阻抗控制到电机空间的映射"功能。

核心依赖：
- J_A(q_s): 驱动雅可比矩阵（2x2），满足 q̇_m = J_A q̇_s 和 τ_s = J_A^T τ_m
- dJ_A/dq_s: J_A对串联关节配置q_s的导数（2x2x2张量）

主要功能：
1. 计算电机空间刚度增益 K_Pm（包含A_Pm、B_Pm、C_Pm三项修正）
2. 计算电机空间阻尼增益 K_Dm
3. 计算串联期望扭矩 τ_s* 并映射为电机期望扭矩 τ_m*
4. 反解电机参考位置 q_m*
"""

import numpy as np
import math
from typing import Tuple, Optional, Union


class SerialImpedanceTransfer:
    """
    串联阻抗传递控制器
    
    实现文档第二节D的核心逻辑，将串联空间的PD阻抗控制映射到电机空间
    """
    
    def __init__(self, 
                 singularity_threshold: float = 1e-6,
                 numerical_step_size: float = 1e-8):
        """
        初始化串联阻抗传递控制器
        
        参数:
            singularity_threshold: 奇异点检测阈值，当|det(J_A)| < threshold时认为奇异
            numerical_step_size: 数值微分的步长（如果使用数值方法计算dJ_A/dq_s）
        """
        self.singularity_threshold = singularity_threshold
        self.numerical_step_size = numerical_step_size
        
        # 状态变量
        self.last_q_s = None
        self.last_J_A = None
        self.last_dJ_A_dq_s = None
        
        print("串联阻抗传递控制器已初始化")
        print(f"奇异点检测阈值: {singularity_threshold}")
        print(f"数值微分步长: {numerical_step_size}")
    
    def compute_motor_stiffness_gain(self, 
                                    J_A: np.ndarray,
                                    dJ_A_dq_s: np.ndarray,
                                    K_Ps: np.ndarray,
                                    q_dot_s: np.ndarray) -> np.ndarray:
        """
        计算电机空间刚度增益 K_Pm
        
        根据文档式26-29，K_Pm = A_Pm + B_Pm + C_Pm
        
        参数:
            J_A: 驱动雅可比矩阵 (2x2)
            dJ_A_dq_s: J_A对q_s的导数 (2x2x2张量)
            K_Ps: 串联空间刚度增益 (2x2对角矩阵)
            q_dot_s: 串联关节角速度 (2x1)
            
        返回:
            K_Pm: 电机空间刚度增益 (2x2)
        """
        try:
            # 检查奇异点
            det_J_A = np.linalg.det(J_A)
            if abs(det_J_A) < self.singularity_threshold:
                raise ValueError(f"雅可比矩阵奇异: det(J_A) = {det_J_A}")
            
            # 计算J_A的逆矩阵
            J_A_inv = np.linalg.inv(J_A)
            J_A_inv_T = J_A_inv.T  # J_A^{-T}
            
            # 计算A_Pm项：A_Pm = J_A^{-T} K_Ps J_A^{-1}
            # 参考文档式26
            A_Pm = J_A_inv_T @ K_Ps @ J_A_inv
            
            # 计算B_Pm项：B_Pm = J_A^{-T} K_Ps (dJ_A/dq_s) q̇_s
            # 参考文档式27
            # dJ_A_dq_s是2x2x2张量，需要与q_dot_s进行张量收缩
            dJ_A_q_dot = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    dJ_A_q_dot[i, j] = np.sum(dJ_A_dq_s[i, j, :] * q_dot_s)
            
            print(f"q_dot_s: {q_dot_s}")
            print(f"dJ_A_q_dot: \n{dJ_A_q_dot}")
            B_Pm = J_A_inv_T @ K_Ps @ dJ_A_q_dot
            
            # 计算C_Pm项：C_Pm = J_A^{-T} K_Ps J_A^{-1} (dJ_A/dq_s) q̇_s
            # 参考文档式28
            C_Pm = J_A_inv_T @ K_Ps @ J_A_inv @ dJ_A_q_dot
            
            # 总刚度增益：K_Pm = A_Pm + B_Pm + C_Pm
            # 参考文档式29
            K_Pm = A_Pm + B_Pm + C_Pm
            print(f"A_Pm + B_Pm + C_Pm: \n{A_Pm}\n+\n{B_Pm}\n+\n{C_Pm}\n=\n{K_Pm}")
            return K_Pm
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"矩阵运算失败: {e}")
    
    def compute_motor_damping_gain(self, 
                                  J_A: np.ndarray,
                                  K_Ds: np.ndarray) -> np.ndarray:
        """
        计算电机空间阻尼增益 K_Dm
        
        根据文档式31：K_Dm = J_A^{-T} K_Ds J_A^{-1}
        
        参数:
            J_A: 驱动雅可比矩阵 (2x2)
            K_Ds: 串联空间阻尼增益 (2x2对角矩阵)
            
        返回:
            K_Dm: 电机空间阻尼增益 (2x2)
        """
        try:
            # 检查奇异点
            det_J_A = np.linalg.det(J_A)
            if abs(det_J_A) < self.singularity_threshold:
                raise ValueError(f"雅可比矩阵奇异: det(J_A) = {det_J_A}")
            
            # 计算J_A的逆矩阵
            J_A_inv = np.linalg.inv(J_A)
            J_A_inv_T = J_A_inv.T  # J_A^{-T}
            
            # 计算K_Dm：K_Dm = J_A^{-T} K_Ds J_A^{-1}
            # 参考文档式31
            K_Dm = J_A_inv_T @ K_Ds @ J_A_inv
            
            return K_Dm
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"矩阵运算失败: {e}")
    
    def compute_serial_desired_torque(self, 
                                    q_s_ref: np.ndarray,
                                    q_s: np.ndarray,
                                    q_dot_s_ref: np.ndarray,
                                    q_dot_s: np.ndarray,
                                    K_Ps: np.ndarray,
                                    K_Ds: np.ndarray) -> np.ndarray:
        """
        计算串联期望扭矩 τ_s*
        
        根据文档式24：τ_s* = K_Ps(q_s* - q_s) + K_Ds(q̇_s* - q̇_s)
        
        参数:
            q_s_ref: 串联关节参考位置 (2x1)
            q_s: 串联关节当前位置 (2x1)
            q_dot_s_ref: 串联关节参考角速度 (2x1)
            q_dot_s: 串联关节当前角速度 (2x1)
            K_Ps: 串联空间刚度增益 (2x2)
            K_Ds: 串联空间阻尼增益 (2x2)
            
        返回:
            tau_s_ref: 串联期望扭矩 (2x1)
        """
        # 计算位置误差和速度误差
        position_error = q_s_ref - q_s
        velocity_error = q_dot_s_ref - q_dot_s
        
        # 计算串联期望扭矩：τ_s* = K_Ps(q_s* - q_s) + K_Ds(q̇_s* - q̇_s)
        # 参考文档式24
        tau_s_ref = K_Ps @ position_error + K_Ds @ velocity_error
        
        return tau_s_ref
    
    def compute_motor_desired_torque(self, 
                                   tau_s_ref: np.ndarray,
                                   J_A: np.ndarray) -> np.ndarray:
        """
        计算电机期望扭矩 τ_m*
        
        根据文档：τ_m* = J_A^{-T} τ_s*
        
        参数:
            tau_s_ref: 串联期望扭矩 (2x1)
            J_A: 驱动雅可比矩阵 (2x2)
            
        返回:
            tau_m_ref: 电机期望扭矩 (2x1)
        """
        try:
            # 检查奇异点
            det_J_A = np.linalg.det(J_A)
            if abs(det_J_A) < self.singularity_threshold:
                raise ValueError(f"雅可比矩阵奇异: det(J_A) = {det_J_A}")
            
            # 计算J_A的逆矩阵
            J_A_inv = np.linalg.inv(J_A)
            J_A_inv_T = J_A_inv.T  # J_A^{-T}
            
            # 计算电机期望扭矩：τ_m* = J_A^{-T} τ_s*
            tau_m_ref = J_A_inv_T @ tau_s_ref
            
            return tau_m_ref
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"矩阵运算失败: {e}")
    
    def compute_motor_reference_position(self, 
                                       tau_m_ref: np.ndarray,
                                       q_m: np.ndarray,
                                       q_dot_m: np.ndarray,
                                       K_Pm: np.ndarray,
                                       K_Dm: np.ndarray) -> np.ndarray:
        """
        计算电机参考位置 q_m*
        
        根据文档式32：q_m* = q_m + K_Pm^{-1}(τ_m* - K_Dm q̇_m)
        
        参数:
            tau_m_ref: 电机期望扭矩 (2x1)
            q_m: 电机当前位置 (2x1)
            q_dot_m: 电机当前角速度 (2x1)
            K_Pm: 电机空间刚度增益 (2x2)
            K_Dm: 电机空间阻尼增益 (2x2)
            
        返回:
            q_m_ref: 电机参考位置 (2x1)
        """
        try:
            # 计算K_Pm的逆矩阵
            K_Pm_inv = np.linalg.inv(K_Pm)
            
            # 计算电机参考位置：q_m* = q_m + K_Pm^{-1}(τ_m* - K_Dm q̇_m)
            # 参考文档式32
            torque_error = tau_m_ref - K_Dm @ q_dot_m
            position_correction = K_Pm_inv @ torque_error
            q_m_ref = q_m + position_correction
            
            return q_m_ref
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"矩阵运算失败: {e}")
    
    def compute_numerical_jacobian_derivative(self, 
                                           J_A_func,
                                           q_s: np.ndarray,
                                           step_size: Optional[float] = None) -> np.ndarray:
        """
        使用数值方法计算J_A对q_s的导数 dJ_A/dq_s
        
        参数:
            J_A_func: 计算J_A(q_s)的函数
            q_s: 串联关节配置 (2x1)
            step_size: 数值微分步长，如果为None则使用默认值
            
        返回:
            dJ_A_dq_s: J_A对q_s的导数 (2x2x2张量)
        """
        if step_size is None:
            step_size = self.numerical_step_size
        
        # 计算当前J_A
        J_A_current = J_A_func(q_s)
        
        # 初始化导数张量
        dJ_A_dq_s = np.zeros((2, 2, 2))
        
        # 对每个q_s分量进行数值微分
        for i in range(2):
            # 创建扰动向量
            delta_q_s = np.zeros(2)
            delta_q_s[i] = step_size
            
            # 计算扰动后的J_A
            J_A_perturbed = J_A_func(q_s + delta_q_s)
            
            # 计算数值导数
            dJ_A_dq_s[:, :, i] = (J_A_perturbed - J_A_current) / step_size
        
        return dJ_A_dq_s
    
    def serial_impedance_transfer(self, 
                                J_A: np.ndarray,
                                dJ_A_dq_s: np.ndarray,
                                K_Ps: np.ndarray,
                                K_Ds: np.ndarray,
                                q_s_ref: np.ndarray,
                                q_s: np.ndarray,
                                q_dot_s_ref: np.ndarray,
                                q_dot_s: np.ndarray,
                                q_m: np.ndarray,
                                q_dot_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        串联阻抗传递主函数
        
        实现完整的串联空间到电机空间的阻抗传递
        
        参数:
            J_A: 驱动雅可比矩阵 (2x2)
            dJ_A_dq_s: J_A对q_s的导数 (2x2x2张量)
            K_Ps: 串联空间刚度增益 (2x2对角矩阵)
            K_Ds: 串联空间阻尼增益 (2x2对角矩阵)
            q_s_ref: 串联关节参考位置 (2x1)
            q_s: 串联关节当前位置 (2x1)
            q_dot_s_ref: 串联关节参考角速度 (2x1)
            q_dot_s: 串联关节当前角速度 (2x1)
            q_m: 电机当前位置 (2x1)
            q_dot_m: 电机当前角速度 (2x1)
            
        返回:
            q_m_ref: 电机参考位置 (2x1)
            K_Pm: 电机空间刚度增益 (2x2)
            K_Dm: 电机空间阻尼增益 (2x2)
        """
        try:
            # 1. 计算电机空间刚度增益 K_Pm
            K_Pm = self.compute_motor_stiffness_gain(J_A, dJ_A_dq_s, K_Ps, q_dot_s)
            
            # 2. 计算电机空间阻尼增益 K_Dm
            K_Dm = self.compute_motor_damping_gain(J_A, K_Ds)
            
            # 3. 计算串联期望扭矩 τ_s*
            tau_s_ref = self.compute_serial_desired_torque(
                q_s_ref, q_s, q_dot_s_ref, q_dot_s, K_Ps, K_Ds
            )
            
            # 4. 计算电机期望扭矩 τ_m*
            tau_m_ref = self.compute_motor_desired_torque(tau_s_ref, J_A)
            
            # 5. 计算电机参考位置 q_m*
            q_m_ref = self.compute_motor_reference_position(
                tau_m_ref, q_m, q_dot_m, K_Pm, K_Dm
            )
            
            return q_m_ref, K_Pm, K_Dm
            
        except Exception as e:
            raise RuntimeError(f"串联阻抗传递计算失败: {e}")
    
    def check_singularity(self, J_A: np.ndarray) -> bool:
        """
        检查雅可比矩阵是否奇异
        
        参数:
            J_A: 驱动雅可比矩阵 (2x2)
            
        返回:
            is_singular: 是否奇异
        """
        det_J_A = np.linalg.det(J_A)
        return abs(det_J_A) < self.singularity_threshold
    
    def get_condition_number(self, J_A: np.ndarray) -> float:
        """
        计算雅可比矩阵的条件数
        
        参数:
            J_A: 驱动雅可比矩阵 (2x2)
            
        返回:
            condition_number: 条件数
        """
        try:
            return np.linalg.cond(J_A)
        except np.linalg.LinAlgError:
            return float('inf')


# 工具函数
def create_diagonal_gain_matrix(kp_values: list, kd_values: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建对角增益矩阵
    
    参数:
        kp_values: 刚度增益值列表 [kp1, kp2]
        kd_values: 阻尼增益值列表 [kd1, kd2]
        
    返回:
        K_Ps: 串联空间刚度增益矩阵 (2x2)
        K_Ds: 串联空间阻尼增益矩阵 (2x2)
    """
    K_Ps = np.diag(kp_values)
    K_Ds = np.diag(kd_values)
    return K_Ps, K_Ds


def validate_input_dimensions(J_A: np.ndarray, 
                             dJ_A_dq_s: np.ndarray,
                             K_Ps: np.ndarray,
                             K_Ds: np.ndarray,
                             q_s_ref: np.ndarray,
                             q_s: np.ndarray,
                             q_dot_s_ref: np.ndarray,
                             q_dot_s: np.ndarray,
                             q_m: np.ndarray,
                             q_dot_m: np.ndarray) -> bool:
    """
    验证输入参数的维度
    
    返回:
        is_valid: 维度是否有效
    """
    try:
        # 检查J_A维度
        assert J_A.shape == (2, 2), f"J_A维度错误: {J_A.shape}, 期望 (2, 2)"
        
        # 检查dJ_A_dq_s维度
        assert dJ_A_dq_s.shape == (2, 2, 2), f"dJ_A_dq_s维度错误: {dJ_A_dq_s.shape}, 期望 (2, 2, 2)"
        
        # 检查增益矩阵维度
        assert K_Ps.shape == (2, 2), f"K_Ps维度错误: {K_Ps.shape}, 期望 (2, 2)"
        assert K_Ds.shape == (2, 2), f"K_Ds维度错误: {K_Ds.shape}, 期望 (2, 2)"
        
        # 检查向量维度
        assert q_s_ref.shape == (2,), f"q_s_ref维度错误: {q_s_ref.shape}, 期望 (2,)"
        assert q_s.shape == (2,), f"q_s维度错误: {q_s.shape}, 期望 (2,)"
        assert q_dot_s_ref.shape == (2,), f"q_dot_s_ref维度错误: {q_dot_s_ref.shape}, 期望 (2,)"
        assert q_dot_s.shape == (2,), f"q_dot_s维度错误: {q_dot_s.shape}, 期望 (2,)"
        assert q_m.shape == (2,), f"q_m维度错误: {q_m.shape}, 期望 (2,)"
        assert q_dot_m.shape == (2,), f"q_dot_m维度错误: {q_dot_m.shape}, 期望 (2,)"
        
        return True
        
    except AssertionError as e:
        print(f"输入维度验证失败: {e}")
        return False


if __name__ == "__main__":
    # 测试代码
    print("=== 串联阻抗传递模块测试 ===")
    
    # 创建控制器
    controller = SerialImpedanceTransfer()
    
    # 创建测试数据
    J_A = np.array([[1.0, 0.1], [0.1, 1.0]])  # 2x2雅可比矩阵
    dJ_A_dq_s = np.random.randn(2, 2, 2) * 0.1  # 2x2x2导数张量
    K_Ps, K_Ds = create_diagonal_gain_matrix([100.0, 100.0], [10.0, 10.0])
    
    q_s_ref = np.array([0.1, 0.05])  # 串联关节参考位置
    q_s = np.array([0.08, 0.06])    # 串联关节当前位置
    q_dot_s_ref = np.array([0.0, 0.0])  # 串联关节参考角速度
    q_dot_s = np.array([0.01, -0.01])  # 串联关节当前角速度
    q_m = np.array([0.5, 0.3])       # 电机当前位置
    q_dot_m = np.array([0.02, 0.01]) # 电机当前角速度
    
    # 验证输入维度
    if validate_input_dimensions(J_A, dJ_A_dq_s, K_Ps, K_Ds, 
                               q_s_ref, q_s, q_dot_s_ref, q_dot_s, q_m, q_dot_m):
        print("✓ 输入维度验证通过")
        
        try:
            # 执行串联阻抗传递
            q_m_ref, K_Pm, K_Dm = controller.serial_impedance_transfer(
                J_A, dJ_A_dq_s, K_Ps, K_Ds,
                q_s_ref, q_s, q_dot_s_ref, q_dot_s, q_m, q_dot_m
            )
            
            print("✓ 串联阻抗传递计算成功")
            print(f"电机参考位置 q_m*: {q_m_ref}")
            print(f"电机刚度增益 K_Pm:\n{K_Pm}")
            print(f"电机阻尼增益 K_Dm:\n{K_Dm}")
            
            # 检查条件数
            condition_number = controller.get_condition_number(J_A)
            print(f"雅可比矩阵条件数: {condition_number:.2e}")
            
        except Exception as e:
            print(f"✗ 串联阻抗传递计算失败: {e}")
    else:
        print("✗ 输入维度验证失败")
