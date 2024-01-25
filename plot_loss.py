import matplotlib.pyplot as plt
import numpy as np

# 假设这是我们从强化学习环境中收集到的回报值列表
rewards = np.load("train_CNN_MLP/CNN_MLP_train_13/reward.npy")

# 计算每十个回报的平均值
average_rewards = [sum(rewards[i:i+50])/50 for i in range(0, len(rewards), 50)]

# 创建一个与average_rewards相对应的episode标记列表
# 每隔十个回报取一个点，所以乘以10
episodes = [(i+1)*50 for i in range(len(average_rewards))]

# 绘制平均回报值
plt.plot(episodes, average_rewards, marker='o', linestyle='-', color='gray')

# 设置图表标题和轴标签
plt.title('Average Reward Every 50 Episodes')
plt.xlabel('Episode')
plt.ylabel('Average Reward')

# 设置横轴刻度，显示每隔十个episode的刻度
# plt.xticks(episodes)

# 开启网格线
plt.grid(True)

# 显示图表
plt.show()