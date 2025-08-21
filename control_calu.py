import socket
import struct
import math
import time

# --- 1. 配置参数 ---
# 请将STM32的IP地址和端口号修改为您的实际配置
STM32_IP = "192.168.1.100"  # 目标STM32的IP地址
UDP_PORT = 9527             # Orin和STM32通信的端口号

# 默认的电机控制参数 (可以根据需要调整)
DEFAULT_KP = 80.0
DEFAULT_KD = 2.0

class MotorController:
    """
    通过UDP控制多路CAN总线上的电机。
    - 封装了数据包的构建和发送。
    - 包含了您提供的最新脚踝逆运动学解算。
    """

    def __init__(self, target_ip, port):
        """
        初始化UDP客户端。
        Args:
            target_ip (str): 目标设备的IP地址。
            port (int): 目标设备的端口号。
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
        print(f"UDP客户端已创建，将向 {target_ip}:{port} 发送数据。")
        print(f"使用的运动学参数: l1={self.l1}, l2={self.l2}, l3={self.l3}, l4={self.l4}, l5={self.l5},r={self.r}")


    def close(self):
        """关闭socket连接。"""
        self.sock.close()
        print("UDP socket已关闭。")

    def _solve_inverse_kinematics(self, theta_p_deg, theta_r_deg):
        """
        【新植入的函数】计算逆向运动学，使用您提供的经过验证的算法。
        内部使用角度制进行计算。
        
        参数:
            theta_p_deg (float): Pitch 角度 (度).
            theta_r_deg (float): Roll 角度 (度).
            
        返回:
            list or None: 成功时返回特定解 [alpha1_deg, alpha2_deg]，失败时返回 None.
        """
        # 直接使用类中定义的参数
        l1, l2, l3, l4, l5,r = self.l1, self.l2, self.l3, self.l4, self.l5,self.r
        
        theta1, theta2 = math.radians(theta_p_deg), math.radians(theta_r_deg)
        c1, s1 = math.cos(theta1), math.sin(theta1)
        c2, s2 = math.cos(theta2), math.sin(theta2)

        try:
            # --- 计算 alpha1 (只取第一个解) ---
            A1 = c2 * l2 + s2 * l5
            B1 = s1 * l1 + c1 * s2 * l2 - c1 * c2 * l5 - l3 + l5
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
            B2 = s1 * l1 - c1 * s2 * l2 - c1 * c2 * l5 - l4 + l5
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

            return [alpha1_sol_deg, alpha2_sol_deg]

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
    controller = MotorController(STM32_IP, UDP_PORT)

    print("正在初始化所有电机，发送位置0指令...")
    # 注意：指令格式为 (位置[转], 力矩[N·m], kp, kd)
    initial_commands = [(0.0, 0.0, 0.0, 0.0)] * 30
    controller.send_command(initial_commands)
    time.sleep(1)
    print("初始化完成。")

    print("\n开始循环控制脚踝电机 (CAN1, 电机1和2)...")
    motor_commands = list(initial_commands) 

    try:
        while True:
            # --- 模拟输入：设定一个变化的pitch和roll角度 (单位: 弧度) ---
            current_time = time.time()
            # 使用正弦波模拟平滑的摇摆动作
            pitch_angle_rad =  20.0/180 *3.14* math.sin(current_time * 1.0)  # 上下摆动
            roll_angle_rad = 12.0/180 *3.14 * math.sin(current_time * 0.7) # 左右摆动 (不同频率以产生更复杂的轨迹)

            # --- 运行新的逆运动学解算 ---
            # 输入是弧度, 输出是转数
            theta_upper, theta_lower = controller.ankle_inverse_kinematics(pitch_angle_rad, roll_angle_rad)
            # 添加零位偏移（与test_inverse_kinematics.py保持一致）
            # theta_upper和theta_lower现在是转数，需要转换为角度再加偏移，最后转回转数
            theta_upper_deg = theta_upper * 360.0  # 转数->角度->加偏移
            theta_lower_deg = theta_lower * 360.0  # 转数->角度->加偏移
            # 转换回转弧度供电机使用
            theta_upper =  math.radians(theta_upper_deg)
            theta_lower =  math.radians(theta_lower_deg)
            # --- 更新脚踝电机的指令 ---
            # 假设: CAN1的电机1是lower, 电机2是upper
            # **注意**: 如果电机运动方向反了，请修改或移除下面的负号
            motor_commands[4] = (theta_lower, 0.0, DEFAULT_KP, DEFAULT_KD) # CAN1, Motor1 (lower)
            motor_commands[5] = (theta_upper, 0.0, DEFAULT_KP, DEFAULT_KD) # CAN1, Motor2 (upper)
            
            # --- 发送更新后的指令 ---
            controller.send_command(motor_commands)

            print(f"Pitch: {math.degrees(pitch_angle_rad):5.1f}°, Roll: {math.degrees(roll_angle_rad):5.1f}° -> Upper: {theta_upper*360/6.28:6.1f}°, Lower: {theta_lower*360/6.28:6.1f}°", end='\r')
            
            # 控制发送频率，例如50Hz
            time.sleep(0.02) 

    except KeyboardInterrupt:
        print("\n检测到Ctrl+C，程序即将退出。")
    finally:
        print("正在发送停止指令...")
        # 将电机复位到0位置
        reset_commands = [(0.0, 0.0, DEFAULT_KP, DEFAULT_KD)] * 30
        controller.send_command(reset_commands)
        time.sleep(0.1)
        controller.close()
        print("程序已安全退出。")
