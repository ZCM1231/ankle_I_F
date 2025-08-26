import numpy as np
import math
# --- 运动学参数定义 (全局) ---
l1, l2, l3, l4, l5 = 50.58, 31.2, 65.0, 127.3, 10.2
# l1, l2, l3, l4, l5 = 0.032, 0.038, 0.066, 0.124, 0.010
# l1, l2, l3, l4, l5 = 50, 38, 66, 124, 10

# ==============================================================================
# 1. 正向运动学求解器 (牛顿法) - 此部分与之前版本相同
# ==============================================================================
def forward_kinematics_newton(alpha1_deg, alpha2_deg, use_analytical_guess=False):
    """
    使用牛顿法计算正向运动学。
    
    参数:
        alpha1_deg (float): 电机1的角度 (度).
        alpha2_deg (float): 电机2的角度 (度).
        use_analytical_guess (bool): 是否使用解析近似解作为初始猜测值。
        
    返回:
        list or None: 成功时返回 [theta1_deg, theta2_deg]，失败时返回 None.
    """

    # --- 初始化 ---
    max_iterations = 100
    tolerance = 1e-5
    alpha1 = math.radians(alpha1_deg)
    alpha2 = math.radians(alpha2_deg)
    ca1, sa1 = math.cos(alpha1), math.sin(alpha1)
    ca2, sa2 = math.cos(alpha2), math.sin(alpha2)

    theta = np.array([0.0, 0.0])

    # --- 牛顿法迭代 ---
    for i in range(max_iterations):
        theta_1, theta_2 = theta[0], theta[1]
        c1, s1 = math.cos(theta_1), math.sin(theta_1)
        c2, s2 = math.cos(theta_2), math.sin(theta_2)

        A = -l1 * c1 + l2 * s1 * s2 - l5 * s1 * c2 + l1 * ca1
        B = l2 * c2 + l5 * s2 - l2
        C = l1 * s1 + l2 * c1 * s2 - l5 * c1 * c2 - l1 * sa1 - l3
        D = A*A + B*B + C*C - l3*l3

        E = -l1 * c1 - l2 * s1 * s2 - l5 * s1 *c2 + l1 * ca2
        F = -l2 * c2 + l5 * s2 + l2
        G = l1 * s1 - l2 * c1 * s2 - l5 * c1 * c2 - l1 * sa2 - l4
        H = E*E + F*F + G*G - l4*l4

        F_vec = np.array([D, H])

        if np.linalg.norm(F_vec) < tolerance:
            return [math.degrees(theta[0]), math.degrees(theta[1])]

        A1 = l1*s1 + l2*s2*c1 - l5*c2*c1
        A2 = l2*s1*c2 + l5*s1*s2
        B1 = 0
        B2 = -l2*s2 + l5*c2
        C1 = l1*c1 - l2*s2*s1 + l5*c2*s1
        C2 = l2*c1*c2 + l5*c1*s2

        E1 = l1*s1 - l2*s2*c1 - l5*c2*c1
        E2 = -l2*s1*c2 + l5*s1*s2
        F1 = 0
        F2 = l2*s2 + l5*c2
        G1 = l1*c1 + l2*s2*s1 + l5*c2*s1
        G2 = -l2*c1*c2 + l5*c1*s2

        J11 = 2*A*A1 + 2*B*B1 + 2*C*C1
        J12 = 2*A*A2 + 2*B*B2 + 2*C*C2
        J21 = 2*E*E1 + 2*F*F1 + 2*G*G1
        J22 = 2*E*E2 + 2*F*F2 + 2*G*G2
        J = np.array([[J11, J12], [J21, J22]])

        try:
            delta = np.linalg.solve(J, -F_vec)
        except np.linalg.LinAlgError:
            return None

        theta += delta

    return None

# ==============================================================================
# 2. 逆向运动学求解器 (已修改)
#    - 根据用户要求，只返回由二次方程“减法解”构成的唯一解
# ==============================================================================
def inverse_kinematics(theta_p_deg, theta_r_deg):
    """
    计算逆向运动学，只返回一个特定的解。
    
    参数:
        theta_p_deg (float): Pitch 角度 (度).
        theta_r_deg (float): Roll 角度 (度).
        
    返回:
        list or None: 成功时返回特定解 [alpha1, alpha2] (度)，失败时返回 None.
    """
    theta1, theta2 = math.radians(theta_p_deg), math.radians(theta_r_deg)
    c1, s1 = math.cos(theta1), math.sin(theta1)
    c2, s2 = math.cos(theta2), math.sin(theta2)

    try:
        # --- 计算 alpha1 (只取第一个解) ---
        A1 = -l1 * c1 + l2 * s1 * s2 - l5 * s1 * c2
        B1 = l1 * s1 + l2 * c1 * s2 - l5 * c1 * c2 - l3
        C1 = (l3**2) - ((l2 * c2) + l5 * s2 - l2)**2
        D1 = (C1 - (A1**2 + B1**2 + l1**2)) / (2 * l1)
        
        a1 = A1 + D1
        b1 = 2 * B1
        c1_coeff = D1 - A1
        
        discriminant1 = b1*b1 - 4*a1*c1_coeff
        if discriminant1 < 0: return None
        
        # 根据要求，只取“减法”解
        t_alpha1 = (-b1 - math.sqrt(discriminant1)) / (2 * a1)
        alpha1_sol = math.degrees(2 * math.atan(t_alpha1))

        # --- 计算 alpha2 (只取第一个解) ---
        A2 = -l1 * c1 - l2 * s1 * s2 - l5 * s1 *c2
        B2 = l1 * s1 - l2 * c1 * s2 - l5 * c1 * c2 - l4
        C2 = (l4**2) - ((-l2 * c2) + l5 * s2 + l2)**2
        D2 = (C2 - (A2**2 + B2**2 + l1**2)) / (2 * l1)

        a2 = A2 + D2
        b2 = 2 * B2
        c2_coeff = D2 - A2
        
        discriminant2 = b2*b2 - 4*a2*c2_coeff
        if discriminant2 < 0: return None

        # 根据要求，只取“减法”解
        t_alpha2 = (-b2 - math.sqrt(discriminant2)) / (2 * a2)
        alpha2_sol = math.degrees(2 * math.atan(t_alpha2))

        return [alpha1_sol, alpha2_sol]

    except (ValueError, ZeroDivisionError):
        # 捕获数学错误（如sqrt负数）或除零错误
        return None
