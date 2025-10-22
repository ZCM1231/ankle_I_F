import socket
import struct
import math
import time
import numpy as np
from serial_impedance_transfer import SerialImpedanceTransfer, create_diagonal_gain_matrix

# --- 1. 配置参数 ---
# 请将STM32的IP地址和端口号修改为您的实际配置
STM32_IP = "192.168.1.100"  # 目标STM32的IP地址
UDP_PORT = 9527             # Orin和STM32通信的端口号

# 默认的串联空间控制参数 (可以根据需要调整)
DEFAULT_KP_SERIAL = 60.0  # 串联空间刚度增益
DEFAULT_KD_SERIAL = 2.0   # 串联空间阻尼增益

class MotorController:
    """
    通过UDP控制多路CAN总线上的电机。
    - 封装了数据包的构建和发送。
    - 包含了您提供的最新脚踝逆运动学解算。
    """

    def __init__(self, target_ip, port, enable_impedance_transfer=True):
        """
        初始化UDP客户端。
        Args:
            target_ip (str): 目标设备的IP地址。
            port (int): 目标设备的端口号。
            enable_impedance_transfer (bool): 是否启用串联阻抗传递功能。
        """
        self.target_addr = (target_ip, port)
        # 创建UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # --- 【已修改】在这里定义统一的运动学参数 ---
        self.l1 = 45.28
        self.l2 = 19.5
        self.l3 = 108.9
        self.l4 = 156.9
        self.l5 = 11
        self.r = 20  
        
        # --- 【新增】串联阻抗传递模块 ---
        self.enable_impedance_transfer = enable_impedance_transfer
        if enable_impedance_transfer:
            self.impedance_transfer = SerialImpedanceTransfer()
            # 默认的串联空间PD增益（使用全局默认值）
            self.K_Ps_default = np.diag([DEFAULT_KP_SERIAL, DEFAULT_KP_SERIAL])  # 串联空间刚度增益
            self.K_Ds_default = np.diag([DEFAULT_KD_SERIAL, DEFAULT_KD_SERIAL])  # 串联空间阻尼增益
            print("✓ 串联阻抗传递模块已启用")
        else:
            self.impedance_transfer = None
            print("⚠ 串联阻抗传递模块已禁用")
        
        print(f"UDP客户端已创建，将向 {target_ip}:{port} 发送数据。")
        print(f"使用的运动学参数: l1={self.l1}, l2={self.l2}, l3={self.l3}, l4={self.l4}, l5={self.l5},r={self.r}")


    def close(self):
        """关闭socket连接。"""
        self.sock.close()
        print("UDP socket已关闭。")

    def _solve_inverse_kinematics(self, theta_p_deg, theta_r_deg):
        """
        计算逆向运动学。
        
        参数:
            theta_p_deg (float): Pitch 角度 (度).
            theta_r_deg (float): Roll 角度 (度).
            
        返回:
            list or None: 成功时返回特定解 [alpha1_deg, alpha2_deg]，失败时返回 None.
        """
        # 直接使用类中定义的参数
        # T1
        l1, l2, l3, l4, l5, r, h3, h4 = 44, 27, 97.82, 145.88, 12 ,30 ,90 ,138
        offest_angle1,offest_angle2 = 0.2630290095, -0.26353345762
        # l1, l2, l3, l4, l5, r, h3, h4 = 44, 27, 159.83, 207.87, 12 ,30 ,155 ,203
        # offest_angle1,offest_angle2 = 0.1614273913, -0.1660209487
        # l1, l2, l3, l4, l5, r, h3, h4 = 45.28, 19.5, 108.9, 156.9, 11 , 20 ,108.9 , 156.9
        # offest_angle1,offest_angle2 = 0, 0
        theta1, theta2 = math.radians(theta_p_deg), math.radians(theta_r_deg)
        c1, s1 = math.cos(theta1), math.sin(theta1)
        c2, s2 = math.cos(theta2), math.sin(theta2)

        try:
            # --- 计算 alpha1 (只取第一个解) ---
            A1 = c2 * l2 + s2 * l5
            B1 = s1 * l1 + c1 * s2 * l2 - c1 * c2 * l5 - h3 + l5
            C1 = l3**2 - (-c1 * l1 + s1 * s2 * l2 - s1 * c2 * l5 + l1)**2

            if math.isclose(l2, 0): return None
            D1 = -(C1 - (A1**2 + B1**2 + r**2)) / (2 * r)
            
            a1 = D1 + A1
            b1 = - 2 * B1
            c1_coeff = D1 - A1
            
            if math.isclose(a1, 0): return None
            
            discriminant1 = b1*b1 - 4*a1*c1_coeff
            if discriminant1 < 0: return None
            
            t_alpha1 = (-b1 + math.sqrt(discriminant1)) / (2 * a1)
            alpha1_sol_deg = math.degrees(2 * math.atan(t_alpha1))

            # --- 计算 alpha2 (只取第一个解) ---
            A2 =  -c2 * l2 + s2 * l5 
            B2 = s1 * l1 - c1 * s2 * l2 - c1 * c2 * l5 - h4 + l5
            C2 = l4**2 - (-c1 * l1 - s1 * s2 * l2 - s1 * c2 * l5 + l1)**2
            
            if math.isclose(l2, 0): return None
            D2 = (C2 - (A2**2 + B2**2 + r**2)) / (2 * r)
            a2 = D2+A2
            b2 =  -2 * B2
            c2_coeff = D2 - A2
            
            if math.isclose(a2, 0): return None

            discriminant2 = b2*b2 - 4*a2*c2_coeff
            if discriminant2 < 0: return None

            t_alpha2 = (-b2 + math.sqrt(discriminant2)) / (2 * a2)
            alpha2_sol_deg = math.degrees(2 * math.atan(t_alpha2))

            return [alpha1_sol_deg-offest_angle1, alpha2_sol_deg-offest_angle2]

        except (ValueError, ZeroDivisionError):
            return None

    # ==============================================================================
    # 【已修改】这是对外暴露的接口，负责单位转换和调用新求解器
    # ==============================================================================
    def ankle_inverse_kinematics(self, theta_p_rad, theta_r_rad):
        """
        【适配层函数】根据给定的脚踝pitch和roll角度（弧度），计算驱动电机的目标位置（转数）。
        它会调用新的、经过验证的逆运动学求解器。

        Args:
            theta_p_rad (float): 脚踝的Pitch角度 (单位: 弧度)。
            theta_r_rad (float): 脚踝的Roll角度 (单位: 弧度)。

        Returns:
            tuple[float, float]: (theta_upper, theta_lower) 电机目标位置(单位: 转数)。
                                 如果无解，则返回 (0.0, 0.0)。
        """
        # 1. 将输入的弧度转换为求解器需要的角度
        theta_p_deg = math.degrees(theta_p_rad)
        theta_r_deg = math.degrees(theta_r_rad)

        # 2. 调用核心求解器
        result_deg = self._solve_inverse_kinematics(theta_p_deg, theta_r_deg)

        # 3. 处理结果
        if result_deg is None:
            # print("警告: 逆运动学无解。")
            return (0.0, 0.0) # 无解时返回0
        
        alpha1_deg, alpha2_deg = result_deg

        # 4. 将求解器返回的角度转换为电机需要的转数
        # 1 转 = 2 * pi 弧度 = 360 度
        alpha1_turns = alpha1_deg / 360.0
        alpha2_turns = alpha2_deg / 360.0

        # --- 假设 alpha2 -> upper, alpha1 -> lower ---
        theta_upper_turns = alpha2_turns
        theta_lower_turns = alpha1_turns
        
        return theta_upper_turns, theta_lower_turns
    
    def compute_actuation_jacobian(self, q_s):
        """
        计算驱动雅可比矩阵 J_A(q_s)
        
        参数:
            q_s: 串联关节配置 [theta_p, theta_r] (弧度)
            
        返回:
            J_A: 驱动雅可比矩阵 (2x2)
        """
        # 使用数值微分计算J_A
        # J_A[i,j] = ∂α_i/∂θ_j，其中α_i是电机角度，θ_j是串联关节角度
        
        h = 1e-6  # 数值微分步长 - 使用更保守的步长
        J_A = np.zeros((2, 2))
        
        # 计算当前逆运动学解
        theta_p_deg, theta_r_deg = math.degrees(q_s[0]), math.degrees(q_s[1])
        result_current = self._solve_inverse_kinematics(theta_p_deg, theta_r_deg)
        
        if result_current is None:
            raise ValueError("当前配置下逆运动学无解")
        
        alpha1_current, alpha2_current = result_current
        
        # 计算 ∂α₁/∂θₚ
        result_dp = self._solve_inverse_kinematics(theta_p_deg + h*180/math.pi, theta_r_deg)
        if result_dp is not None:
            J_A[0, 0] = (result_dp[0] - alpha1_current) / h
        
        # 计算 ∂α₁/∂θᵣ
        result_dr = self._solve_inverse_kinematics(theta_p_deg, theta_r_deg + h*180/math.pi)
        if result_dr is not None:
            J_A[0, 1] = (result_dr[0] - alpha1_current) / h
        
        # 计算 ∂α₂/∂θₚ
        if result_dp is not None:
            J_A[1, 0] = (result_dp[1] - alpha2_current) / h
        
        # 计算 ∂α₂/∂θᵣ
        if result_dr is not None:
            J_A[1, 1] = (result_dr[1] - alpha2_current) / h
        
        # 转换为弧度制
        J_A = J_A * math.pi / 180
        
        return J_A
    
    def compute_jacobian_derivative(self, q_s):
        """
        计算J_A对q_s的导数 dJ_A/dq_s
        
        参数:
            q_s: 串联关节配置 [theta_p, theta_r] (弧度)
            
        返回:
            dJ_A_dq_s: J_A对q_s的导数 (2x2x2张量)
        """
        h = 1e-6  # 数值微分步长 - 使用更保守的步长
        dJ_A_dq_s = np.zeros((2, 2, 2))
        
        # 计算当前J_A
        J_A_current = self.compute_actuation_jacobian(q_s)
        
        # 对每个q_s分量进行数值微分
        for i in range(2):
            # 创建扰动向量
            delta_q_s = np.zeros(2)
            delta_q_s[i] = h
            
            try:
                # 计算扰动后的J_A
                J_A_perturbed = self.compute_actuation_jacobian(q_s + delta_q_s)
                
                # 计算数值导数
                dJ_A_dq_s[:, :, i] = (J_A_perturbed - J_A_current) / h
                print(f"dJ_A/dq_s[{i}]: \n{dJ_A_dq_s[:, :, i]}")
            except ValueError:
                # 如果扰动后无解，使用零矩阵
                dJ_A_dq_s[:, :, i] = np.zeros((2, 2))
        
        return dJ_A_dq_s
    
    def ankle_impedance_control(self, 
                              q_s_ref, q_s, q_dot_s_ref, q_dot_s,
                              q_m, q_dot_m,
                              K_Ps=None, K_Ds=None):
        """
        脚踝阻抗控制主函数
        
        实现串联空间阻抗控制到电机空间的映射
        
        参数:
            q_s_ref: 串联关节参考位置 [theta_p_ref, theta_r_ref] (弧度)
            q_s: 串联关节当前位置 [theta_p, theta_r] (弧度)
            q_dot_s_ref: 串联关节参考角速度 [theta_p_dot_ref, theta_r_dot_ref] (弧度/秒)
            q_dot_s: 串联关节当前角速度 [theta_p_dot, theta_r_dot] (弧度/秒)
            q_m: 电机当前位置 [alpha1, alpha2] (弧度)
            q_dot_m: 电机当前角速度 [alpha1_dot, alpha2_dot] (弧度/秒)
            K_Ps: 串联空间刚度增益矩阵 (2x2)，如果为None则使用默认值
            K_Ds: 串联空间阻尼增益矩阵 (2x2)，如果为None则使用默认值
            
        返回:
            q_m_ref: 电机参考位置 [alpha1_ref, alpha2_ref] (弧度)
            K_Pm: 电机空间刚度增益矩阵 (2x2)
            K_Dm: 电机空间阻尼增益矩阵 (2x2)
        """
        if not self.enable_impedance_transfer:
            raise RuntimeError("串联阻抗传递模块未启用")
        
        # 使用默认增益矩阵（如果未提供）
        if K_Ps is None:
            K_Ps = self.K_Ps_default
        if K_Ds is None:
            K_Ds = self.K_Ds_default
        
        try:
            # 计算驱动雅可比矩阵
            J_A = self.compute_actuation_jacobian(q_s)
            
            # 计算J_A的导数
            dJ_A_dq_s = self.compute_jacobian_derivative(q_s)
            
            # 执行串联阻抗传递
            q_m_ref, K_Pm, K_Dm = self.impedance_transfer.serial_impedance_transfer(
                J_A, dJ_A_dq_s, K_Ps, K_Ds,
                q_s_ref, q_s, q_dot_s_ref, q_dot_s,
                q_m, q_dot_m
            )
            
            return q_m_ref, K_Pm, K_Dm
            
        except Exception as e:
            print(f"脚踝阻抗控制计算失败: {e}")
            # 返回当前位置作为参考位置（保持当前位置）
            return q_m.copy(), np.eye(2) * 100.0, np.eye(2) * 10.0

    def _create_motor_frame(self, position, torque, kp, kd):
        """创建单个电机的10字节指令帧。"""
        pos_cmd = int(position * 10000.0)
        vel_cmd = 0
        tor_cmd = int((torque + 0.03313) / 0.004938)
        kp_cmd = int(kp * 6.28 * 10 / 0.4938)
        kd_cmd = int(kd * 6.28 * 10 / 0.4938)
        frame = struct.pack('<hhhhh', pos_cmd, vel_cmd, tor_cmd, kp_cmd, kd_cmd)
        return frame

    def _build_udp_packet(self, motor_commands):
        """构建完整的UDP数据包。"""
        if len(motor_commands) != 30:
            raise ValueError("需要提供全部30个电机的指令！")
        packet = bytearray(b'\x95\x27\x02')
        cmd_len = 320
        packet.extend(struct.pack('<H', cmd_len))
        for can_id in range(1, 6):
            can_data = bytearray(64)
            for motor_id_on_can in range(1, 7):
                motor_idx = (can_id - 1) * 6 + (motor_id_on_can - 1)
                cmd = motor_commands[motor_idx]
                motor_frame = self._create_motor_frame(*cmd)
                start_index = 10 * (motor_id_on_can - 1)
                can_data[start_index : start_index + 10] = motor_frame
            can_data[62] = 0x17
            can_data[63] = 0x01
            packet.extend(can_data)
        check_sum = (packet[2] + cmd_len + packet[-1]) & 0xFF
        packet.append(check_sum)
        return bytes(packet)

    def send_command(self, motor_commands):
        """构建并发送电机指令。"""
        try:
            udp_packet = self._build_udp_packet(motor_commands)
            self.sock.sendto(udp_packet, self.target_addr)
        except Exception as e:
            print(f"发送数据时出错: {e}")


if __name__ == "__main__":
    # 创建控制器，启用串联阻抗传递功能
    controller = MotorController(STM32_IP, UDP_PORT, enable_impedance_transfer=True)

    print("正在初始化所有电机，发送位置0指令...")
    # 注意：指令格式为 (位置[转], 力矩[N·m], kp, kd)
    initial_commands = [(0.0, 0.0, 0.0, 0.0)] * 30
    controller.send_command(initial_commands)
    time.sleep(1)
    print("初始化完成。")

    print("\n开始循环控制脚踝电机 (CAN1, 电机1和2)...")
    print("模式选择:")
    print("1. 传统逆运动学控制")
    print("2. 串联阻抗传递控制")
    
    # 选择控制模式
    control_mode = 2  # 默认使用阻抗控制模式
    
    motor_commands = list(initial_commands) 
    
    # 初始化状态变量
    q_m_current = np.array([0.0, 0.0])  # 电机当前位置
    q_dot_m_current = np.array([0.0, 0.0])  # 电机当前角速度
    last_q_m_current = np.array([0.0, 0.0])  # 上一时刻的电机当前位置
    
    # 如果是阻抗模式，需要初始化电机角度
    if control_mode == 2:
        try:
            # 计算零位下的电机角度
            theta_upper_turns, theta_lower_turns = controller.ankle_inverse_kinematics(0.0, 0.0)
            if theta_upper_turns != 0.0 or theta_lower_turns != 0.0:
                alpha1_init = theta_lower_turns * 2 * math.pi
                alpha2_init = theta_upper_turns * 2 * math.pi
                q_m_current = np.array([alpha1_init, alpha2_init])
                last_q_m_current = q_m_current.copy()
                print(f"阻抗模式初始化：α₁={math.degrees(alpha1_init):.1f}°, α₂={math.degrees(alpha2_init):.1f}°")
        except Exception as e:
            print(f"阻抗模式初始化失败: {e}")

    try:
        while True:
            # --- 模拟输入：设定一个变化的pitch和roll角度 (单位: 弧度) ---
            current_time = time.time()
            # 使用正弦波模拟平滑的摇摆动作
            pitch_angle_rad =  15.0/180 *3.14* math.sin(current_time * 1.0)  # 上下摆动
            # roll_angle_rad = 12.0/180 *3.14 * math.sin(current_time * 0.7) # 左右摆动 (不同频率以产生更复杂的轨迹)
            # pitch_angle_rad =  15.0/180 *3.14  # 上下摆动
            roll_angle_rad = 0.0/180 *3.14 # 左右摆动 (不同频率以产生更复杂的轨迹)
            
            # 串联关节状态
            q_s_ref = np.array([pitch_angle_rad, roll_angle_rad])  # 串联关节参考位置
            q_s = q_s_ref.copy()  # 假设当前位置等于参考位置（理想情况）
            q_dot_s_ref = np.array([0.0, 0.0])  # 串联关节参考角速度
            
            # 为了测试B_Pm和C_Pm，使用非零速度
            # 在实际应用中，这些速度应该来自传感器或状态估计
            q_dot_s = np.array([0.1, 0.1])  # 串联关节当前角速度（测试用）

            if control_mode == 1:
                # --- 传统逆运动学控制 ---
                theta_upper, theta_lower = controller.ankle_inverse_kinematics(pitch_angle_rad, roll_angle_rad)
                theta_upper_deg = theta_upper * 360.0
                theta_lower_deg = theta_lower * 360.0
                theta_upper = math.radians(theta_upper_deg)
                theta_lower = math.radians(theta_lower_deg)
                
                # 更新电机指令
                motor_commands[4] = (theta_lower, 0.0, DEFAULT_KP_SERIAL, DEFAULT_KD_SERIAL) # CAN1, Motor1 (lower)
                motor_commands[5] = (theta_upper, 0.0, DEFAULT_KP_SERIAL, DEFAULT_KD_SERIAL) # CAN1, Motor2 (upper)
                
                print(f"[传统模式] Pitch: {math.degrees(pitch_angle_rad):5.1f}°, Roll: {math.degrees(roll_angle_rad):5.1f}° -> Upper: {theta_upper*360/6.28:6.1f}°, Lower: {theta_lower*360/6.28:6.1f}°", end='\r')
                
            elif control_mode == 2:
                # --- 串联阻抗传递控制 ---
                try:
                    # 1. 先通过逆运动学计算当前电机角度
                    theta_upper_turns, theta_lower_turns = controller.ankle_inverse_kinematics(pitch_angle_rad, roll_angle_rad)
                    
                    if theta_upper_turns == 0.0 and theta_lower_turns == 0.0:
                        # 逆运动学无解，回退到传统模式
                        raise ValueError("逆运动学无解")
                    
                    # 2. 将转数转换为弧度
                    alpha1_current = theta_lower_turns * 2 * math.pi  # lower motor -> alpha1
                    alpha2_current = theta_upper_turns * 2 * math.pi  # upper motor -> alpha2
                    q_m_current = np.array([alpha1_current, alpha2_current])
                    
                    # 3. 执行串联阻抗传递（只计算增益，不计算位置）
                    _, K_Pm, K_Dm = controller.ankle_impedance_control(
                        q_s_ref, q_s, q_dot_s_ref, q_dot_s,
                        q_m_current, q_dot_m_current
                    )
                    
                    # 4. 更新电机指令（使用传统逆运动学的位置 + 阻抗传递的增益）
                    # 位置使用传统逆运动学的结果
                    alpha1_ref_turns = theta_lower_turns  # lower motor
                    alpha2_ref_turns = theta_upper_turns  # upper motor
                    
                    # 增益使用阻抗传递的结果
                    kp_motor1 = K_Pm[0, 0]  # lower motor 刚度增益
                    kd_motor1 = K_Dm[0, 0]  # lower motor 阻尼增益
                    kp_motor2 = K_Pm[1, 1]  # upper motor 刚度增益
                    kd_motor2 = K_Dm[1, 1]  # upper motor 阻尼增益
                    
                    # 更新电机指令
                    motor_commands[4] = (alpha1_current, 0.0, kp_motor1, kd_motor1) # CAN1, Motor1 (lower)
                    motor_commands[5] = (alpha2_current, 0.0, kp_motor2, kd_motor2) # CAN1, Motor2 (upper)
                    
                    # 更新状态变量（使用传统逆运动学的位置）
                    q_m_current = np.array([alpha1_current, alpha2_current])
                    q_dot_m_current = (q_m_current - last_q_m_current) / 0.02  # 简化的速度估计
                    last_q_m_current = q_m_current.copy()
                    
                    print(f"[阻抗模式] Pitch: {math.degrees(pitch_angle_rad):5.1f}°, Roll: {math.degrees(roll_angle_rad):5.1f}° -> α₁: {math.degrees(alpha1_current):6.1f}°, α₂: {math.degrees(alpha2_current):6.1f}° | KP: [{kp_motor1:.2f}, {kp_motor2:.2f}], KD: [{kd_motor1:.2f}, {kd_motor2:.2f}]", end='\r')
                    
                except Exception as e:
                    print(f"[阻抗模式] 计算失败: {e}")
                    # 回退到传统模式
                    theta_upper, theta_lower = controller.ankle_inverse_kinematics(pitch_angle_rad, roll_angle_rad)
                    theta_upper_deg = theta_upper * 360.0
                    theta_lower_deg = theta_lower * 360.0
                    theta_upper = math.radians(theta_upper_deg)
                    theta_lower = math.radians(theta_lower_deg)
                    
                    motor_commands[4] = (theta_lower, 0.0, DEFAULT_KP_SERIAL, DEFAULT_KD_SERIAL)
                    motor_commands[5] = (theta_upper, 0.0, DEFAULT_KP_SERIAL, DEFAULT_KD_SERIAL)
            
            # --- 发送更新后的指令 ---
            controller.send_command(motor_commands)
            
            # 控制发送频率，例如50Hz
            time.sleep(0.02) 

    except KeyboardInterrupt:
        print("\n检测到Ctrl+C，程序即将退出。")
    finally:
        print("正在发送停止指令...")
        # 将电机复位到0位置
        reset_commands = [(0.0, 0.0, DEFAULT_KP_SERIAL, DEFAULT_KD_SERIAL)]
        controller.send_command(reset_commands)
        time.sleep(0.1)
        controller.close()
        print("程序已安全退出。")
