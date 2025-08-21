import numpy as np
import math
def inverse_kinematics(theta_p_deg, theta_r_deg):
    """
    计算逆向运动学。
    
    参数:
        theta_p_deg (float): Pitch 角度 (度).
        theta_r_deg (float): Roll 角度 (度).
        
    返回:
        list or None: 成功时返回特定解 [alpha1_deg, alpha2_deg]，失败时返回 None.
    """
    # 直接使用类中定义的参数
    l1, l2, l3, l4,l5 ,r ,h3 ,h4 =  44, 22.5, 159.83, 207.87, 12 ,30 ,155 ,203
    offest_angle1,offest_angle2 = 9.26/180. * math.pi, -9.34/180. * math.pi
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

        return [alpha1_sol_deg, alpha2_sol_deg]

    except (ValueError, ZeroDivisionError):
        return None
def forward_kinematics_constraints(theta_p, theta_r, alpha1, alpha2):
    """
    2DOF踝关节并联机构的约束方程
    """
    # 机构参数
    l1, l2, l3, l4,l5 ,r ,h3 ,h4 =  44, 22.5, 159.83, 207.87, 12 ,30 ,155 ,203
    offest_angle1,offest_angle2 = 9.26/180. * math.pi, -9.34/180. * math.pi
    # 角度转换
    cp, sp = math.cos(theta_p), math.sin(theta_p)
    cr, sr = math.cos(theta_r), math.sin(theta_r)
    ca1, sa1 = math.cos(alpha1), math.sin(alpha1)
    ca2, sa2 = math.cos(alpha2), math.sin(alpha2)
    
    # 末端平台旋转矩阵：先绕Y轴转pitch，再绕X轴转roll
    # R = Rx(θᵣ) * Ry(θₚ)
    R = np.array([
        [cp,      sp*sr,     sp*cr],
        [0,       cr,       -sr   ],
        [-sp,     cp*sr,     cp*cr]
    ])
    
    # === 末端平台上的点经过旋转后的位置 ===
    A_orig = np.array([-l1, l2, -l5])
    D_orig = np.array([-l1, -l2, -l5])
    
    A_new = R @ A_orig  # A旋转后的位置
    D_new = R @ D_orig  # D旋转后的位置
    
    # === 驱动关节的位置 ===
    # B绕b点在yz平面内旋转α₁角度
    b_pos = np.array([-l1, 0, -l5+h3])
    # B初始时相对于b的位置是[0, r, 0]
    B_rel = np.array([0, r*ca1, r*sa1])
    B_new = b_pos + B_rel
    
    # C绕c点在yz平面内旋转α₂角度  
    c_pos = np.array([-l1, 0, -l5+h4])
    # C初始时相对于c的位置是[0, -r, 0]
    C_rel = np.array([0, -r*ca2, -r*sa2])
    C_new = c_pos + C_rel
    
    # === 约束方程 ===
    # 约束1：AB连杆长度恒定
    AB_vector = B_new - A_new
    AB_length_squared = np.dot(AB_vector, AB_vector)
    # 计算AB的原始长度
    AB_orig_vector = np.array([-l1, r*math.cos(offest_angle1), -l5+h3+r*math.sin(offest_angle1)]) - np.array([-l1, l2, -l5])
    # AB_orig_length_squared = np.dot(AB_orig_vector, AB_orig_vector)
    AB_orig_length_squared = l3 ** 2
    constraint1 = AB_length_squared - AB_orig_length_squared
    
    # 约束2：CD连杆长度恒定
    CD_vector = C_new - D_new
    CD_length_squared = np.dot(CD_vector, CD_vector)
    # 计算CD的原始长度
    CD_orig_vector = np.array([-l1, -r*math.cos(offest_angle2), -l5+h4-r*math.sin(offest_angle2)]) - np.array([-l1, -l2, -l5])
    # CD_orig_length_squared = np.dot(CD_orig_vector, CD_orig_vector)
    CD_orig_length_squared = l4 ** 2
    
    constraint2 = CD_length_squared - CD_orig_length_squared
    
    return np.array([constraint1, constraint2])

def compute_jacobian(theta_p, theta_r, alpha1, alpha2):
    """
    计算约束方程的雅可比矩阵
    """
    # 数值微分计算雅可比矩阵
    h = 1e-8  # 微小增量
    F0 = forward_kinematics_constraints(theta_p, theta_r, alpha1, alpha2)
    
    # ∂F/∂θₚ
    F_dp = forward_kinematics_constraints(theta_p + h, theta_r, alpha1, alpha2)
    dF_dp = (F_dp - F0) / h
    
    # ∂F/∂θᵣ
    F_dr = forward_kinematics_constraints(theta_p, theta_r + h, alpha1, alpha2)
    dF_dr = (F_dr - F0) / h
    
    J = np.column_stack([dF_dp, dF_dr])
    return J

def forward_kinematics_newton(alpha1_deg, alpha2_deg, initial_guess=None, max_iter=100, tol=1e-4):
    """
    使用牛顿法求解正运动学
    """
    # 转换为弧度
    alpha1 = math.radians(alpha1_deg)
    alpha2 = math.radians(alpha2_deg)
    
    # 初始猜测
    if initial_guess is None:
        theta = np.array([0.0, 0.0])  # [theta_p, theta_r] in radians
    else:
        theta = np.array([math.radians(initial_guess[0]), math.radians(initial_guess[1])])
    
    # print(f"目标: α₁={alpha1_deg:.2f}°, α₂={alpha2_deg:.2f}°")
    # print(f"初始猜测: θₚ={math.degrees(theta[0]):.2f}°, θᵣ={math.degrees(theta[1]):.2f}°")
    # print("-" * 60)
    
    for i in range(max_iter):
        theta_p, theta_r = theta[0], theta[1]
        
        # 计算约束函数值
        F = forward_kinematics_constraints(theta_p, theta_r, alpha1, alpha2)
        
        # print(f"迭代 {i+1}:")
        # print(f"  θₚ={math.degrees(theta_p):8.4f}°, θᵣ={math.degrees(theta_r):8.4f}°")
        # print(f"  F₁={F[0]:12.8f}, F₂={F[1]:12.8f}")
        # print(f"  ||F||={np.linalg.norm(F):12.8f}")
        
        # 检查收敛
        if np.linalg.norm(F) < tol:
            # print("✓ 收敛成功!")
            return [math.degrees(theta_p), math.degrees(theta_r)]
        
        # 计算雅可比矩阵
        J = compute_jacobian(theta_p, theta_r, alpha1, alpha2)
        
        # print(f"  J = [{J[0,0]:10.6f} {J[0,1]:10.6f}]")
        # print(f"      [{J[1,0]:10.6f} {J[1,1]:10.6f}]")
        
        # 检查奇异性
        det_J = np.linalg.det(J)
        # print(f"  det(J) = {det_J:.8f}")
        
        if abs(det_J) < 1e-12:
            # print("✗ 雅可比矩阵奇异")
            return None
        
        # 牛顿步
        try:
            delta_theta = np.linalg.solve(J, -F)
            # print(f"  Δθₚ={math.degrees(delta_theta[0]):8.4f}°, Δθᵣ={math.degrees(delta_theta[1]):8.4f}°")
            
            # 更新
            theta += delta_theta
            
        except np.linalg.LinAlgError:
            # print("✗ 线性求解失败")
            return None
        
        # print()
    
    # print("✗ 达到最大迭代次数")
    return None
