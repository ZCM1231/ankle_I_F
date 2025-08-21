import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from I_F import forward_kinematics_newton, inverse_kinematics
# def inverse_kinematics(theta_p_deg, theta_r_deg):
#     """
#     计算逆向运动学。
    
#     参数:
#         theta_p_deg (float): Pitch 角度 (度).
#         theta_r_deg (float): Roll 角度 (度).
        
#     返回:
#         list or None: 成功时返回特定解 [alpha1_deg, alpha2_deg]，失败时返回 None.
#     """
#     # 直接使用类中定义的参数
#     l1, l2, l3, l4,l5 ,r ,h3 ,h4 =  44, 22.5, 159.83, 207.87, 12 ,30 ,155 ,203
#     offest_angle1,offest_angle2 = 9.26/180. * math.pi, -9.34/180. * math.pi
#     theta1, theta2 = math.radians(theta_p_deg), math.radians(theta_r_deg)
#     c1, s1 = math.cos(theta1), math.sin(theta1)
#     c2, s2 = math.cos(theta2), math.sin(theta2)

#     try:
#         # --- 计算 alpha1 (只取第一个解) ---
#         A1 = c2 * l2 + s2 * l5
#         B1 = s1 * l1 + c1 * s2 * l2 - c1 * c2 * l5 - h3 + l5
#         C1 = l3**2 - (-c1 * l1 + s1 * s2 * l2 - s1 * c2 * l5 + l1)**2

#         if math.isclose(l2, 0): return None
#         D1 = -(C1 - (A1**2 + B1**2 + r**2)) / (2 * r)
        
#         a1 = D1 + A1
#         b1 = - 2 * B1
#         c1_coeff = D1 - A1
        
#         if math.isclose(a1, 0): return None
        
#         discriminant1 = b1*b1 - 4*a1*c1_coeff
#         if discriminant1 < 0: return None
        
#         t_alpha1 = (-b1 + math.sqrt(discriminant1)) / (2 * a1)
#         alpha1_sol_deg = math.degrees(2 * math.atan(t_alpha1))

#         # --- 计算 alpha2 (只取第一个解) ---
#         A2 =  -c2 * l2 + s2 * l5 
#         B2 = s1 * l1 - c1 * s2 * l2 - c1 * c2 * l5 - h4 + l5
#         C2 = l4**2 - (-c1 * l1 - s1 * s2 * l2 - s1 * c2 * l5 + l1)**2
        
#         if math.isclose(l2, 0): return None
#         D2 = (C2 - (A2**2 + B2**2 + r**2)) / (2 * r)
#         a2 = D2+A2
#         b2 =  -2 * B2
#         c2_coeff = D2 - A2
        
#         if math.isclose(a2, 0): return None

#         discriminant2 = b2*b2 - 4*a2*c2_coeff
#         if discriminant2 < 0: return None

#         t_alpha2 = (-b2 + math.sqrt(discriminant2)) / (2 * a2)
#         alpha2_sol_deg = math.degrees(2 * math.atan(t_alpha2))

#         return [alpha1_sol_deg, alpha2_sol_deg]

#     except (ValueError, ZeroDivisionError):
#         return None
# def forward_kinematics_constraints(theta_p, theta_r, alpha1, alpha2):
#     """
#     2DOF踝关节并联机构的约束方程
#     """
#     # 机构参数
#     l1, l2, l3, l4,l5 ,r ,h3 ,h4 =  44, 22.5, 159.83, 207.87, 12 ,30 ,155 ,203
#     offest_angle1,offest_angle2 = 9.26/180. * math.pi, -9.34/180. * math.pi
#     # 角度转换
#     cp, sp = math.cos(theta_p), math.sin(theta_p)
#     cr, sr = math.cos(theta_r), math.sin(theta_r)
#     ca1, sa1 = math.cos(alpha1), math.sin(alpha1)
#     ca2, sa2 = math.cos(alpha2), math.sin(alpha2)
    
#     # 末端平台旋转矩阵：先绕Y轴转pitch，再绕X轴转roll
#     # R = Rx(θᵣ) * Ry(θₚ)
#     R = np.array([
#         [cp,      sp*sr,     sp*cr],
#         [0,       cr,       -sr   ],
#         [-sp,     cp*sr,     cp*cr]
#     ])
    
#     # === 末端平台上的点经过旋转后的位置 ===
#     A_orig = np.array([-l1, l2, -l5])
#     D_orig = np.array([-l1, -l2, -l5])
    
#     A_new = R @ A_orig  # A旋转后的位置
#     D_new = R @ D_orig  # D旋转后的位置
    
#     # === 驱动关节的位置 ===
#     # B绕b点在yz平面内旋转α₁角度
#     b_pos = np.array([-l1, 0, -l5+h3])
#     # B初始时相对于b的位置是[0, r, 0]
#     B_rel = np.array([0, r*ca1, r*sa1])
#     B_new = b_pos + B_rel
    
#     # C绕c点在yz平面内旋转α₂角度  
#     c_pos = np.array([-l1, 0, -l5+h4])
#     # C初始时相对于c的位置是[0, -r, 0]
#     C_rel = np.array([0, -r*ca2, -r*sa2])
#     C_new = c_pos + C_rel
    
#     # === 约束方程 ===
#     # 约束1：AB连杆长度恒定
#     AB_vector = B_new - A_new
#     AB_length_squared = np.dot(AB_vector, AB_vector)
#     # 计算AB的原始长度
#     AB_orig_vector = np.array([-l1, r*math.cos(offest_angle1), -l5+h3+r*math.sin(offest_angle1)]) - np.array([-l1, l2, -l5])
#     AB_orig_length_squared = np.dot(AB_orig_vector, AB_orig_vector)
    
#     constraint1 = AB_length_squared - AB_orig_length_squared
    
#     # 约束2：CD连杆长度恒定
#     CD_vector = C_new - D_new
#     CD_length_squared = np.dot(CD_vector, CD_vector)
#     # 计算CD的原始长度
#     CD_orig_vector = np.array([-l1, -r*math.cos(offest_angle2), -l5+h4-r*math.sin(offest_angle2)]) - np.array([-l1, -l2, -l5])
#     CD_orig_length_squared = np.dot(CD_orig_vector, CD_orig_vector)
    
#     constraint2 = CD_length_squared - CD_orig_length_squared
    
#     return np.array([constraint1, constraint2])

# def compute_jacobian(theta_p, theta_r, alpha1, alpha2):
#     """
#     计算约束方程的雅可比矩阵
#     """
#     # 数值微分计算雅可比矩阵
#     h = 1e-8
#     F0 = forward_kinematics_constraints(theta_p, theta_r, alpha1, alpha2)
    
#     # ∂F/∂θₚ
#     F_dp = forward_kinematics_constraints(theta_p + h, theta_r, alpha1, alpha2)
#     dF_dp = (F_dp - F0) / h
    
#     # ∂F/∂θᵣ
#     F_dr = forward_kinematics_constraints(theta_p, theta_r + h, alpha1, alpha2)
#     dF_dr = (F_dr - F0) / h
    
#     J = np.column_stack([dF_dp, dF_dr])
#     return J

# def forward_kinematics_newton(alpha1_deg, alpha2_deg, initial_guess=None, max_iter=100, tol=1e-2):
#     """
#     使用牛顿法求解正运动学
#     """
#     # 转换为弧度
#     alpha1 = math.radians(alpha1_deg)
#     alpha2 = math.radians(alpha2_deg)
    
#     # 初始猜测
#     if initial_guess is None:
#         theta = np.array([0.0, 0.0])  # [theta_p, theta_r] in radians
#     else:
#         theta = np.array([math.radians(initial_guess[0]), math.radians(initial_guess[1])])
    
#     # print(f"目标: α₁={alpha1_deg:.2f}°, α₂={alpha2_deg:.2f}°")
#     # print(f"初始猜测: θₚ={math.degrees(theta[0]):.2f}°, θᵣ={math.degrees(theta[1]):.2f}°")
#     # print("-" * 60)
    
#     for i in range(max_iter):
#         theta_p, theta_r = theta[0], theta[1]
        
#         # 计算约束函数值
#         F = forward_kinematics_constraints(theta_p, theta_r, alpha1, alpha2)
        
#         # print(f"迭代 {i+1}:")
#         # print(f"  θₚ={math.degrees(theta_p):8.4f}°, θᵣ={math.degrees(theta_r):8.4f}°")
#         # print(f"  F₁={F[0]:12.8f}, F₂={F[1]:12.8f}")
#         # print(f"  ||F||={np.linalg.norm(F):12.8f}")
        
#         # 检查收敛
#         if np.linalg.norm(F) < tol:
#             # print("✓ 收敛成功!")
#             return [math.degrees(theta_p), math.degrees(theta_r)]
        
#         # 计算雅可比矩阵
#         J = compute_jacobian(theta_p, theta_r, alpha1, alpha2)
        
#         # print(f"  J = [{J[0,0]:10.6f} {J[0,1]:10.6f}]")
#         # print(f"      [{J[1,0]:10.6f} {J[1,1]:10.6f}]")
        
#         # 检查奇异性
#         det_J = np.linalg.det(J)
#         # print(f"  det(J) = {det_J:.8f}")
        
#         if abs(det_J) < 1e-12:
#             # print("✗ 雅可比矩阵奇异")
#             return None
        
#         # 牛顿步
#         try:
#             delta_theta = np.linalg.solve(J, -F)
#             # print(f"  Δθₚ={math.degrees(delta_theta[0]):8.4f}°, Δθᵣ={math.degrees(delta_theta[1]):8.4f}°")
            
#             # 更新
#             theta += delta_theta
            
#         except np.linalg.LinAlgError:
#             # print("✗ 线性求解失败")
#             return None
        
#         # print()
    
#     # print("✗ 达到最大迭代次数")
#     return None
def test_with_meshgrid():
    """
    使用mesh grid创建系统性测试用例
    """
    print("=== 使用Mesh Grid进行全工作空间测试 ===\n")
    
    # 定义测试范围
    theta_p_range = np.linspace(-45, 27, 300)  
    theta_r_range = np.linspace(-15, 15, 300)  
    # 创建mesh grid
    THETA_P, THETA_R = np.meshgrid(theta_p_range, theta_r_range)
    
    # 初始化结果存储
    results = {
        'theta_p': [],
        'theta_r': [],
        'alpha1': [],
        'alpha2': [],
        'theta_p_calc': [],
        'theta_r_calc': [],
        'error_p': [],
        'error_r': [],
        'max_error': [],
        'inverse_success': [],
        'forward_success': [],
        'overall_success': []
    }
    
    total_tests = THETA_P.size
    inverse_failures = 0
    forward_failures = 0
    high_error_count = 0
    
    print(f"总测试点数: {total_tests}")
    print(f"测试范围: θₚ ∈ [{theta_p_range[0]:.1f}°, {theta_p_range[-1]:.1f}°]")
    print(f"测试范围: θᵣ ∈ [{theta_r_range[0]:.1f}°, {theta_r_range[-1]:.1f}°]")
    print("-" * 60)
    
    # 遍历所有测试点
    for i in range(THETA_P.shape[0]):
        for j in range(THETA_P.shape[1]):
            theta_p_target = THETA_P[i, j]
            theta_r_target = THETA_R[i, j]
            
            # 记录目标值
            results['theta_p'].append(theta_p_target)
            results['theta_r'].append(theta_r_target)
            
            # 逆运动学求解
            alpha_result = inverse_kinematics(theta_p_target, theta_r_target)
            
            if alpha_result is None:
                # 逆运动学失败
                inverse_failures += 1
                results['alpha1'].append(np.nan)
                results['alpha2'].append(np.nan)
                results['theta_p_calc'].append(np.nan)
                results['theta_r_calc'].append(np.nan)
                results['error_p'].append(np.nan)
                results['error_r'].append(np.nan)
                results['max_error'].append(np.nan)
                results['inverse_success'].append(False)
                results['forward_success'].append(False)
                results['overall_success'].append(False)
                continue
            
            alpha1, alpha2 = alpha_result
            results['alpha1'].append(alpha1)
            results['alpha2'].append(alpha2)
            results['inverse_success'].append(True)
            
            # 正运动学求解
            forward_result = forward_kinematics_newton(
                alpha1, alpha2, 
                initial_guess=[theta_p_target * 0.8, theta_r_target * 0.8],
                max_iter=20,
                tol=1e-2
            )
            
            if forward_result is None:
                # 正运动学失败
                forward_failures += 1
                results['theta_p_calc'].append(np.nan)
                results['theta_r_calc'].append(np.nan)
                results['error_p'].append(np.nan)
                results['error_r'].append(np.nan)
                results['max_error'].append(np.nan)
                results['forward_success'].append(False)
                results['overall_success'].append(False)
                continue
            
            theta_p_calc, theta_r_calc = forward_result
            results['theta_p_calc'].append(theta_p_calc)
            results['theta_r_calc'].append(theta_r_calc)
            results['forward_success'].append(True)
            
            # 计算误差
            error_p = abs(theta_p_target - theta_p_calc)
            error_r = abs(theta_r_target - theta_r_calc)
            max_error = max(error_p, error_r)
            
            results['error_p'].append(error_p)
            results['error_r'].append(error_r)
            results['max_error'].append(max_error)
            
            # 判断是否成功（误差小于0.01°）
            success = max_error < 0.5
            results['overall_success'].append(success)
            
            if not success:
                high_error_count += 1
    
    # 转换为numpy数组以便分析
    for key in results:
        results[key] = np.array(results[key])
    
    # 统计结果
    successful_tests = np.sum(results['overall_success'])
    success_rate = successful_tests / total_tests * 100
    
    print(f"\n=== 测试统计结果 ===")
    print(f"总测试点数: {total_tests}")
    print(f"逆运动学失败: {inverse_failures} ({inverse_failures/total_tests*100:.1f}%)")
    print(f"正运动学失败: {forward_failures} ({forward_failures/total_tests*100:.1f}%)")
    print(f"高误差点数: {high_error_count} ({high_error_count/total_tests*100:.1f}%)")
    print(f"成功测试点数: {successful_tests} ({success_rate:.1f}%)")
    
    # 误差统计（仅针对成功的测试）
    valid_errors = results['max_error'][~np.isnan(results['max_error'])]
    if len(valid_errors) > 0:
        print(f"\n=== 误差统计（成功测试点）===")
        print(f"平均误差: {np.mean(valid_errors):.6f}°")
        print(f"最大误差: {np.max(valid_errors):.6f}°")
        print(f"误差标准差: {np.std(valid_errors):.6f}°")
        print(f"99%分位数: {np.percentile(valid_errors, 99):.6f}°")
    
    return results, THETA_P, THETA_R

def visualize_results(results, THETA_P, THETA_R):
    """
    可视化测试结果
    """
    # 重新整理数据为mesh grid格式
    success_grid = np.array(results['overall_success']).reshape(THETA_P.shape).T
    error_grid = np.array(results['max_error']).reshape(THETA_P.shape).T
    
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 成功/失败分布图
    plt.subplot(2, 3, 1)
    plt.imshow(success_grid, extent=[THETA_R.min(), THETA_R.max(), 
                                   THETA_P.min(), THETA_P.max()], 
               origin='lower', cmap='RdYlGn', aspect='auto')
    plt.colorbar(label='Success (1) / Failure (0)')
    plt.xlabel('Roll θᵣ (°)')
    plt.ylabel('Pitch θₚ (°)')
    plt.title('Success/Failure Distribution')
    plt.grid(True, alpha=0.3)
    
    # 2. 误差分布热图
    plt.subplot(2, 3, 2)
    error_plot = np.where(np.isnan(error_grid), -1, error_grid)  # NaN用-1表示
    im = plt.imshow(error_plot, extent=[THETA_R.min(), THETA_R.max(), 
                                      THETA_P.min(), THETA_P.max()], 
                   origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Max Error (°)')
    plt.xlabel('Roll θᵣ (°)')
    plt.ylabel('Pitch θₚ (°)')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # 3. 误差直方图
    plt.subplot(2, 3, 3)
    valid_errors = results['max_error'][~np.isnan(results['max_error'])]
    if len(valid_errors) > 0:
        plt.hist(valid_errors, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(valid_errors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(valid_errors):.6f}°')
        plt.axvline(np.median(valid_errors), color='blue', linestyle='--', 
                   label=f'Median: {np.median(valid_errors):.6f}°')
        plt.xlabel('Max Error (°)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution Histogram')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. 3D误差表面
    ax = fig.add_subplot(2, 3, 4, projection='3d')
    # 只绘制成功的点
    mask = ~np.isnan(error_grid)
    if np.any(mask):
        surf = ax.plot_surface(THETA_R, THETA_P, np.where(mask, error_grid, 0), 
                              cmap='viridis', alpha=0.8)
        ax.set_xlabel('Roll θᵣ (°)')
        ax.set_ylabel('Pitch θₚ (°)')
        ax.set_zlabel('Max Error (°)')
        ax.set_title('3D Error Surface')
        plt.colorbar(surf, ax=ax, shrink=0.5)
    
    # 5. 逆运动学成功率
    plt.subplot(2, 3, 5)
    inverse_success_grid = np.array(results['inverse_success']).reshape(THETA_P.shape).T
    plt.imshow(inverse_success_grid, extent=[THETA_R.min(), THETA_R.max(), 
                                           THETA_P.min(), THETA_P.max()], 
               origin='lower', cmap='RdYlGn', aspect='auto')
    plt.colorbar(label='Inverse Success (1) / Failure (0)')
    plt.xlabel('Roll θᵣ (°)')
    plt.ylabel('Pitch θₚ (°)')
    plt.title('Inverse Kinematics Success Rate')
    plt.grid(True, alpha=0.3)
    
    # 6. 工作空间边界
    plt.subplot(2, 3, 6)
    # 绘制成功点
    success_mask = np.array(results['overall_success'])
    theta_p_success = np.array(results['theta_p'])[success_mask]
    theta_r_success = np.array(results['theta_r'])[success_mask]
    
    # 绘制失败点
    fail_mask = ~success_mask
    theta_p_fail = np.array(results['theta_p'])[fail_mask]
    theta_r_fail = np.array(results['theta_r'])[fail_mask]
    
    plt.scatter(theta_r_success, theta_p_success, c='green', s=20, alpha=0.6, label='Success')
    if len(theta_p_fail) > 0:
        plt.scatter(theta_r_fail, theta_p_fail, c='red', s=20, alpha=0.6, label='Failure')
    
    plt.xlabel('Roll θᵣ (°)')
    plt.ylabel('Pitch θₚ (°)')
    plt.title('Workspace Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def analyze_workspace_boundary(results):
    """
    分析工作空间边界
    """
    print("\n=== 工作空间边界分析 ===")
    
    success_mask = np.array(results['overall_success'])
    theta_p_success = np.array(results['theta_p'])[success_mask]
    theta_r_success = np.array(results['theta_r'])[success_mask]
    
    if len(theta_p_success) > 0:
        print(f"Pitch工作范围: [{np.min(theta_p_success):.1f}°, {np.max(theta_p_success):.1f}°]")
        print(f"Roll工作范围: [{np.min(theta_r_success):.1f}°, {np.max(theta_r_success):.1f}°]")
        
        # 计算有效工作空间面积（近似）
        pitch_range = np.max(theta_p_success) - np.min(theta_p_success)
        roll_range = np.max(theta_r_success) - np.min(theta_r_success)
        theoretical_area = pitch_range * roll_range
        actual_points = len(theta_p_success)
        total_points = len(results['theta_p'])
        workspace_efficiency = actual_points / total_points * 100
        
        print(f"理论工作空间: {theoretical_area:.1f} deg²")
        print(f"实际工作空间效率: {workspace_efficiency:.1f}%")

def detailed_meshgrid_test():
    """
    执行详细的mesh grid测试
    """
    # 执行测试
    results, THETA_P, THETA_R = test_with_meshgrid()
    
    # 分析工作空间
    analyze_workspace_boundary(results)
    
    # 可视化结果
    visualize_results(results, THETA_P, THETA_R)
    
    # 查找问题区域
    print("\n=== 问题区域分析 ===")
    
    # 找出高误差点
    high_error_mask = np.array(results['max_error']) > 0.5
    high_error_mask = high_error_mask & ~np.isnan(np.array(results['max_error']))
    
    if np.any(high_error_mask):
        print("高误差点 (>0.5°):")
        high_error_indices = np.where(high_error_mask)[0]
        for idx in high_error_indices[:10]:  # 只显示前10个
            print(f"  θₚ={results['theta_p'][idx]:6.1f}°, θᵣ={results['theta_r'][idx]:6.1f}°, "
                  f"误差={results['max_error'][idx]:.6f}°")
        if len(high_error_indices) > 10:
            print(f"  ... 还有 {len(high_error_indices)-10} 个高误差点")
    
    # 找出逆运动学失败点
    inverse_fail_mask = ~np.array(results['inverse_success'])
    if np.any(inverse_fail_mask):
        print("\n逆运动学失败点:")
        fail_indices = np.where(inverse_fail_mask)[0]
        for idx in fail_indices[:10]:  # 只显示前10个
            print(f"  θₚ={results['theta_p'][idx]:6.1f}°, θᵣ={results['theta_r'][idx]:6.1f}°")
        if len(fail_indices) > 10:
            print(f"  ... 还有 {len(fail_indices)-10} 个失败点")
    
    return results

# 运行完整测试
if __name__ == "__main__":
    # 需要先导入之前的函数
    results = detailed_meshgrid_test()