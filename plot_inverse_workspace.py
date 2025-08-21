import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from tqdm import tqdm

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
    l1, l2, l3, l4, l5 , r ,h3 ,h4=  45.28, 19.5, 108.9, 156.9, 11 , 20 ,108.9 , 156.9
    
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

if __name__ == "__main__":
    # 角度范围设置
    pitch_range_deg = 60  # Pitch角度范围±24度
    roll_range_deg = 60  # Roll角度范围±14度
    resolution = 500  # 提高分辨率以匹配目标区域分析的精度
    
    print(f"分析范围: Pitch [{-pitch_range_deg}, {pitch_range_deg}]°, Roll [{-roll_range_deg}, {roll_range_deg}]°")
    print(f"分辨率: {resolution}x{resolution}")

    # 创建角度网格
    pitch_axis = np.linspace(-pitch_range_deg, pitch_range_deg, resolution)
    roll_axis = np.linspace(-roll_range_deg, roll_range_deg, resolution)
    
    # 创建网格并初始化工作空间地图
    workspace_map = np.zeros((resolution, resolution))
    unreachable_points = []

    print(f"开始分析 {resolution}x{resolution} 的工作空间...")
    
    # 遍历每个点进行分析
    for i in tqdm(range(resolution)):
        pitch_deg = pitch_axis[i]
        for j in range(resolution):
            roll_deg = roll_axis[j]
            
            solution = inverse_kinematics(pitch_deg, roll_deg)
            
            is_valid_solution = (solution is not None and 
                               all(math.isfinite(val) for val in solution))
            
            if is_valid_solution:
                workspace_map[i, j] = 1
            else:
                workspace_map[i, j] = 0
                unreachable_points.append((pitch_deg, roll_deg))

    # 计算覆盖率
    target_reachable_count = np.sum(workspace_map)
    target_total_count = workspace_map.size
    target_coverage = (target_reachable_count / target_total_count) * 100
    print(f"\n目标区域覆盖率: {target_coverage:.1f}% ({target_reachable_count}/{target_total_count} 个点可达)")
    
    # 显示不可达位置
    if unreachable_points:
        print(f"\n不可达位置 (共 {len(unreachable_points)} 个):")
        for i, (pitch, roll) in enumerate(unreachable_points):
            print(f"  {i+1:2d}. Pitch: {pitch:6.1f}°, Roll: {roll:6.1f}°")
            if i >= 19:  # 限制显示前20个，避免输出过长
                remaining = len(unreachable_points) - 20
                if remaining > 0:
                    print(f"  ... 还有 {remaining} 个不可达点未显示")
                break
    else:
        print("\n所有位置均可达!")

    # --- 可视化结果 ---
    print("\n正在生成并保存图像...")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 设置坐标轴范围
    ax.set_xlim(-roll_range_deg, roll_range_deg)
    ax.set_ylim(-pitch_range_deg, pitch_range_deg)
    
    # 设置网格
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_xticks(np.arange(-roll_range_deg, roll_range_deg + 1, 5))
    ax.set_yticks(np.arange(-pitch_range_deg, pitch_range_deg + 1, 5))
    
    # 创建目标区域矩形
    rect_boundary = Rectangle(
        xy=(-roll_range_deg, -pitch_range_deg),
        width=2 * roll_range_deg,
        height=2 * pitch_range_deg,
        edgecolor='green',
        facecolor='none',
        linewidth=2,
        linestyle='-',
        label=f'目标区域 [Pitch: ±{pitch_range_deg}°, Roll: ±{roll_range_deg}°] - 覆盖率: {target_coverage:.1f}%'
    )
    ax.add_patch(rect_boundary)
    
    # 在图像上标记不可达点
    if unreachable_points:
        unreachable_rolls = [point[1] for point in unreachable_points]
        unreachable_pitches = [point[0] for point in unreachable_points]
        ax.scatter(unreachable_rolls, unreachable_pitches, 
                  c='red', marker='.', s=1, alpha=0.3, 
                  label=f'不可达点 ({len(unreachable_points)}个)')
    
    ax.set_title('逆运动学工作空间分析', fontsize=18, pad=15)
    ax.set_xlabel('Roll 角度 [度]', fontsize=14)
    ax.set_ylabel('Pitch 角度 [度]', fontsize=14)
    
    # 创建图例
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    output_filename = 'inverse_kinematics_workspace_with_boundary.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"图像已成功保存为: {output_filename}")
    plt.show()
