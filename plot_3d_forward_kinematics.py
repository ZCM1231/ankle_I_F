import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from I_F import forward_kinematics_newton

# 创建motor输入的网格
alpha1_range = np.linspace(-80, 80, 200)  # motor 1的角度范围
alpha2_range = np.linspace(-80, 80, 200)  # motor 2的角度范围
alpha1_grid, alpha2_grid = np.meshgrid(alpha1_range, alpha2_range)

# 初始化结果数组
pitch_grid = np.zeros_like(alpha1_grid)
roll_grid = np.zeros_like(alpha1_grid)

# 计算每个点的pitch和roll
for i in range(len(alpha1_range)):
    for j in range(len(alpha2_range)):
        alpha1 = alpha1_grid[i, j]
        alpha2 = alpha2_grid[i, j]
        
        # 使用正向运动学计算pitch和roll
        result = forward_kinematics_newton(alpha1, alpha2, initial_guess=[0, 0])
        
        if result is not None:
            pitch_grid[i, j] = result[0]  # pitch
            roll_grid[i, j] = result[1]   # roll
        else:
            pitch_grid[i, j] = np.nan
            roll_grid[i, j] = np.nan

# 设置合理的数值范围限制
pitch_limit = 45  # 限制pitch角度在±45度范围内
roll_limit = 45   # 限制roll角度在±45度范围内

# 将超出范围的值设置为nan
pitch_grid[(pitch_grid > pitch_limit) | (pitch_grid < -pitch_limit)] = np.nan
roll_grid[(roll_grid > roll_limit) | (roll_grid < -roll_limit)] = np.nan

# 创建两个子图
fig = plt.figure(figsize=(15, 6))

# Pitch图
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(alpha1_grid, alpha2_grid, pitch_grid, cmap='viridis', 
                        linewidth=0, antialiased=True)
ax1.set_xlabel('Motor 1 angle(degree)')
ax1.set_ylabel('Motor 2 angle(degree)')
ax1.set_zlabel('Pitch angle(degree)')
ax1.set_title('Motor input and Pitch')
ax1.set_zlim(-pitch_limit, pitch_limit)  # 设置z轴范围
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Roll图
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(alpha1_grid, alpha2_grid, roll_grid, cmap='viridis',
                        linewidth=0, antialiased=True)
ax2.set_xlabel('Motor 1 angle(degree)')
ax2.set_ylabel('Motor 2 angle(degree)')
ax2.set_zlabel('Roll angle(degree)')
ax2.set_title('Motor input and Roll')
ax2.set_zlim(-roll_limit, roll_limit)  # 设置z轴范围
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()
