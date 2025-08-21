import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from I_F import inverse_kinematics

# 创建pitch和roll输入的网格
pitch_range = np.linspace(-45, 25, 200)  # pitch角度范围
roll_range = np.linspace(-15, 15, 200)   # roll角度范围
pitch_grid, roll_grid = np.meshgrid(pitch_range, roll_range)

# 初始化结果数组
motor1_grid = np.zeros_like(pitch_grid)
motor2_grid = np.zeros_like(pitch_grid)

# 计算每个点的motor角度
for i in range(len(pitch_range)):
    for j in range(len(roll_range)):
        pitch = pitch_grid[i, j]
        roll = roll_grid[i, j]
        
        # 使用逆向运动学计算motor角度
        result = inverse_kinematics(pitch, roll)
        
        if result is not None:
            motor1_grid[i, j] = result[0]  # motor1
            motor2_grid[i, j] = result[1]  # motor2
        else:
            motor1_grid[i, j] = np.nan
            motor2_grid[i, j] = np.nan

# 创建两个子图
fig = plt.figure(figsize=(15, 6))

# Motor1图
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(pitch_grid, roll_grid, motor1_grid, cmap='viridis')
ax1.set_xlabel('Pitch angle(degree)')
ax1.set_ylabel('Roll angle(degree)')
ax1.set_zlabel('Motor 1 angle(degree)')
ax1.set_title('Pitch/Roll and Motor1')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Motor2图
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(pitch_grid, roll_grid, motor2_grid, cmap='viridis')
ax2.set_xlabel('Pitch angle(degree)')
ax2.set_ylabel('Roll angle(degree)')
ax2.set_zlabel('Motor 2 angle(degree)')
ax2.set_title('Pitch/Roll and Motor2')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()
