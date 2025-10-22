"""
并联腿机构数学建模与运动学分析
================================

本文档详细描述了2DOF踝关节并联机构的数学建模、逆运动学和正运动学分析方法。

1. 并联腿数学建模
================

建模参数共有十个，以T0.5为例子分别为：
    l1, l2, l3, l4, l5, r, h3, h4 = 44, 22.5, 159.83, 207.87, 12, 30, 155, 203
    offest_angle1, offest_angle2 = 9.26°, -9.34°

其中：
- l1, l2, l3, l4, l5, r: 各个杆长和驱动杆圆周径长
- h3, h4: 踝关节到驱动电机的高度
- offest_angle1, offest_angle2: 电机连杆相对于水平面的初始角度

坐标系建立：
- 将踝关节的旋转中心作为建系的零点O(0, 0, 0)
- 短连杆AB，长连杆CD
- 下电机驱动点B绕OB旋转，上电机驱动点C绕OC旋转

在pitch=0°, roll=0°时，各个点的坐标为：
- O(0, 0, 0)
- A(-l1, l2, -l5)
- B(-l1, r·cos(offest_angle1), h3 + r·sin(offest_angle1) - l5)
- C(-l1, -r·cos(offest_angle1), h4 + r·sin(offest_angle1) - l5)
- D(-l1, -l2, -l5)

当相对于0°旋转角度pitch=θ₁，roll=θ₂时，下电机旋转角度α₁，上电机旋转角度α₂，
可以计算ABCD在旋转后的三维坐标A'B'C'D'：

点A'的变换矩阵：
设c₁=cos(θ₁), s₁=sin(θ₁), c₂=cos(θ₂), s₂=sin(θ₂)
首先计算两个旋转矩阵的乘积 Rot_y(θ₁) * Rot_x(θ₂) * 平移:

[ c₁,  0,  s₁,  0 ]   [ 1,   0,   0,  0 ]   [ 1,  0,  0, -l1 ]
[  0,  1,   0,  0 ] * [ 0,  c₂, -s₂,  0 ] * [ 0,  1,  0,  l2 ]
[-s₁,  0,  c₁,  0 ]   [ 0,  s₂,  c₂,  0 ]   [ 0,  0,  1, -l5 ]
[  0,  0,   0,  1 ]   [ 0,   0,   0,  1 ]   [ 0,  0,  0,   1 ]

T = [         c₁                s₁s₂                   s₁c₂     -c₁l₁ + s₁s₂l₂ - s₁c₂l₅ ]
     [         0                  c₂                     -s₂             c₂l₂ + s₂l₅      ]
     [        -s₁                c₁s₂                   c₁c₂     s₁l₁ + c₁s₂l₂ - c₁c₂l₅  ]
     [         0                  0                      0                 1              ]

A'(-c₁l₁ + s₁s₂l₂ - s₁c₂l₅, c₂l₂ + s₂l₅, s₁l₁ + c₁s₂l₂ - c₁c₂l₅)

点B'的变换矩阵：
设sa₁=sin(α₁+offest_angle1), ca₁=cos(α₁+offest_angle1)

[ 1  0  0  -l1  ]    [1   0   0   0 ]     [ 1  0  0  0   ]
[ 0  1  0  0    ] *  [ 0  ca₁  sa₁  0 ]  *  [ 0  1  0   r ]
[ 0  0  1  h3-l5]    [0  -sa₁  ca₁  0 ]     [ 0  0  1   0  ]
[ 0  0  0  1    ]    [ 0   0  0   1 ]     [ 0  0  0   1  ]

T = [ 1  0  0  -l1     ]
     [ 0  ca₁  0  r·ca₁ ]
     [ 0  -sa₁  1  -r·sa₁+h3-l5]
     [ 0  0  0  1     ]

B'(-l1, r·ca₁, -r·sa₁+h3-l5)

点C'的变换矩阵：
T = [         c₁                s₁s₂                   s₁c₂     -c₁l₁ - s₁s₂l₂ - s₁c₂l₅ ]
     [         0                  c₂                     -s₂             -c₂l₂ + s₂l₅      ]
     [        -s₁                c₁s₂                   c₁c₂     s₁l₁ - c₁s₂l₂ - c₁c₂l₅  ]
     [         0                  0                      0                 1              ]

C'(-c₁l₁ - s₁s₂l₂ - s₁c₂l₅, -c₂l₂ + s₂l₅, s₁l₁ - c₁s₂l₂ - c₁c₂l₅)

点D'的变换矩阵：
设sa₂=sin(α₂+offest_angle2), ca₂=cos(α₂+offest_angle2)

T = [ 1  0  0  -l1     ]
     [ 0  ca₂  0  -r·ca₂ ]
     [ 0  -sa₂  1  r·sa₂+h4-l5]
     [ 0  0  0  1     ]

D'(-l1, -r·ca₂, r·sa₂+h4-l5)

2. 并联腿逆运动学
================

约束的建立：
根据连杆约束，当pitch和roll已知时，|A'B'|=l₃，|C'D'|=l₄，
那么有两组方程可以用来对未知量α₁，α₂求解，这就是并联腿的逆运动学解析解。

|A'B'|=l₃ 的约束方程：
(-c₁l₁+s₁s₂l₂-s₁c₂l₅+l₁)² + (c₂l₂+s₂l₅-r·ca₁)² + (s₁l₁+c₁s₂l₂-c₁c₂l₅+r·sa₁-h₃+l₅)² = l₃²

整理得到：
(c₂l₂+s₂l₅-r·ca₁)² + (s₁l₁+c₁s₂l₂-c₁c₂l₅+r·sa₁-h₃+l₅)² = l₃² - (-c₁l₁+s₁s₂l₂-s₁c₂l₅+l₁)²

假设：
A₁ = c₂l₂ + s₂l₅
B₁ = s₁l₁ + c₁s₂l₂ - c₁c₂l₅ - h₃ + l₅
C₁ = l₃² - (-c₁l₁ + s₁s₂l₂ - s₁c₂l₅ + l₁)²

则约束方程变为：
(A₁ - r·ca₁)² + (B₁ + r·sa₁)² = C₁

展开得到：
r² + A₁² + B₁² - 2A₁r·ca₁ + 2B₁r·sa₁ = C₁

整理得到：
B₁·sa₁ - A₁·ca₁ = (C₁ - (r² + A₁² + B₁²)) / (2r)

设D₁ = (C₁ - (r² + A₁² + B₁²)) / (2r)
则：B₁·sa₁ - A₁·ca₁ = D₁

设t₁ = tan(α₁/2)，利用半角公式：
2t₁B₁/(1+t₁²) + (1-t₁²)A₁/(1+t₁²) = D₁

整理得到：
(A₁ + D₁)t₁² - 2B₁t₁ + (D₁ - A₁) = 0

求解得到：
t₁ = (2B₁ ± √((2B₁)² - 4(A₁ + D₁)(D₁ - A₁))) / (2(A₁ + D₁))

同理，对于α₂：
t₂ = (2B₂ ± √((2B₂)² - 4(A₂ + D₂)(D₂ - A₂))) / (2(A₂ + D₂))

其中：
A₂ = -c₂l₂ + s₂l₅
B₂ = s₁l₁ - c₁s₂l₂ - c₁c₂l₅ - h₄ + l₅
C₂ = l₄² - (-c₁l₁ - s₁s₂l₂ - s₁c₂l₅ + l₁)²
D₂ = (C₂ - (A₂² + B₂² + r²)) / (2r)

解的选取：
该解析方法求解时存在多解问题。选定解的方向的方法为：
根据建模方式，将解出来的解带入数学模型当中，以此来确定解的方向是否合理。

如选取减法解时，模型出现了连杆交叠问题，而取加法解的时候，连杆没有任何交叠，
故选取加法解。

3. 并联腿正运动学
================

3.1 正运动学问题描述
正运动学是指已知驱动关节角度α₁和α₂，求解末端平台的位置和姿态（pitch和roll角度）。
这是一个典型的并联机构正运动学问题，需要通过约束方程求解。

3.2 约束方程建立
根据连杆约束条件，当驱动角度α₁和α₂已知时，需要满足以下两个约束方程：

约束1：AB连杆长度恒定
|A'B'| = l₃

约束2：CD连杆长度恒定
|C'D'| = l₄

其中：
- A'是末端平台上的点A经过旋转后的位置
- B'是下电机驱动点B的位置
- C'是上电机驱动点C的位置
- D'是末端平台上的点D经过旋转后的位置

3.3 约束方程数学表达
基于前面的数学建模，约束方程可以表示为：

约束1：
(-c₁l₁ + s₁s₂l₂ - s₁c₂l₅ + l₁)² + (c₂l₂ + s₂l₅ - r·cos(α₁ + offset_angle₁))² + 
(s₁l₁ + c₁s₂l₂ - c₁c₂l₅ + r·sin(α₁ + offset_angle₁) - h₃ + l₅)² = l₃²

约束2：
(-c₁l₁ - s₁s₂l₂ - s₁c₂l₅ + l₁)² + (-c₂l₂ + s₂l₅ + r·cos(α₂ + offset_angle₂))² + 
(s₁l₁ - c₁s₂l₂ - c₁c₂l₅ - r·sin(α₂ + offset_angle₂) - h₄ + l₅)² = l₄²

其中：
- c₁ = cos(θ₁), s₁ = sin(θ₁) (pitch角度)
- c₂ = cos(θ₂), s₂ = sin(θ₂) (roll角度)
- θ₁, θ₂是待求解的未知量

3.4 数值求解方法
由于约束方程是非线性的，无法直接获得解析解，因此采用数值方法求解。
代码中实现了牛顿-拉夫森法：

牛顿法迭代公式：
θ^(k+1) = θ^(k) - J^(-1) · F(θ^(k))

其中：
- θ = [θ₁, θ₂]是待求解的角度向量
- F(θ) = [F₁, F₂]是约束函数向量
- J是雅可比矩阵

雅可比矩阵通过数值微分计算：
J = [∂F₁/∂θ₁  ∂F₁/∂θ₂]
    [∂F₂/∂θ₁  ∂F₂/∂θ₂]

其中：
∂Fᵢ/∂θⱼ ≈ (Fᵢ(θ + h·eⱼ) - Fᵢ(θ)) / h

h是微小增量，eⱼ是第j个分量的单位向量。

3.5 求解流程
1. 初始化：给定驱动角度α₁和α₂，设定初始猜测值θ₀ = [0°, 0°]
2. 迭代求解：
   - 计算约束函数值F(θ^(k))
   - 计算雅可比矩阵J(θ^(k))
   - 求解线性方程组J·Δθ = -F
   - 更新解：θ^(k+1) = θ^(k) + Δθ
3. 收敛判断：当||F(θ)|| < ε时认为收敛
4. 奇异性检查：当|det(J)| < δ时认为雅可比矩阵奇异

3.6 解的选取与验证
与逆运动学类似，正运动学也可能存在多解问题。通过以下方式确保解的合理性：
1. 物理约束：解必须在机构的物理工作空间内
2. 连续性：相邻时刻的解应该连续变化
3. 连杆约束：确保连杆长度满足设计要求

3.7 算法特点
- 收敛性：牛顿法具有二次收敛特性，收敛速度快
- 稳定性：通过奇异性检查提高数值稳定性
- 精度：可调节收敛容差，平衡计算精度和效率
- 鲁棒性：设置最大迭代次数，避免无限循环

这种基于约束方程的数值求解方法为并联腿机构的正运动学提供了有效的解决方案，
能够处理复杂的非线性约束关系。

"""

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
    # l1, l2, l3, l4, l5, r, h3, h4 = 44, 27, 97.82, 145.88, 12 ,30 ,90 ,138
    # offest_angle1,offest_angle2 = 0.2630290095, -0.26353345762
    l1, l2, l3, l4, l5, r, h3, h4 = 44, 27, 159.83, 207.87, 12 ,30 ,155 ,203
    offest_angle1,offest_angle2 = 0.1614273913, -0.1660209487
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
def forward_kinematics_constraints(theta_p, theta_r, alpha1, alpha2):
    """
    2DOF踝关节并联机构的约束方程
    """
    # 机构参数
    l1, l2, l3, l4, l5, r, h3, h4 = 44, 27, 97.82, 145.88, 12 ,30 ,90 ,138
    offest_angle1,offest_angle2 = 0.2630290095, -0.26353345762
    # l1, l2, l3, l4, l5, r, h3, h4 = 44, 22.5, 159.83, 207.87, 12 ,30 ,155 ,203
    # offest_angle1,offest_angle2 = 0.1563259276, -0.1589336433
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

def compute_kinematic_jacobian(theta_p, theta_r):
    """
    计算运动学雅可比矩阵 - 描述末端执行器角速度与关节速度的关系
    
    参数:
        theta_p: pitch角度（弧度）
        theta_r: roll角度（弧度）
        
    返回:
        J: 2x2运动学雅可比矩阵，满足 [θ̇₁] = J [α̇₁]
                                    [θ̇₂]    [α̇₂]
    """
    h = 1e-12  # 数值微分的步长
    
    # 计算当前构型下的逆运动学解
    theta_p_deg = math.degrees(theta_p)
    theta_r_deg = math.degrees(theta_r)
    result = inverse_kinematics(theta_p_deg, theta_r_deg)
    if result is None:
        return None
    alpha1_deg, alpha2_deg = result
    
    # 计算逆运动学雅可比矩阵的四个元素（数值微分）
    J_inv = np.zeros((2, 2))
    
    # ∂α₁/∂θ₁
    result_dp = inverse_kinematics(theta_p_deg + h*180/math.pi, theta_r_deg)
    if result_dp is None:
        return None
    J_inv[0,0] = (result_dp[0] - alpha1_deg) / h
    
    # ∂α₁/∂θ₂
    result_dr = inverse_kinematics(theta_p_deg, theta_r_deg + h*180/math.pi)
    if result_dr is None:
        return None
    J_inv[0,1] = (result_dr[0] - alpha1_deg) / h
    
    # ∂α₂/∂θ₁
    J_inv[1,0] = (result_dp[1] - alpha2_deg) / h
    
    # ∂α₂/∂θ₂
    J_inv[1,1] = (result_dr[1] - alpha2_deg) / h
    # 转换为弧度制的雅可比矩阵
    J_inv = J_inv * math.pi / 180
    
    # 求逆得到运动学雅可比矩阵
    try:
        J = np.linalg.inv(J_inv)
        return J
    except np.linalg.LinAlgError:
        return None

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

if __name__ == "__main__":
    # 计算零位姿态下的运动学雅可比矩阵
    J = compute_kinematic_jacobian(0, 0)
    print("零位姿态下的运动学雅可比矩阵:")
    print(J)
    if J is not None:
        # 计算J*J^T及其逆
        JJT = J @ J.T
        JJT_inv = np.linalg.inv(JJT)
        # print("\nJ*J^T:")
        # print(JJT)
        print("\n(J*J^T)^(-1):")
        print(JJT_inv)
