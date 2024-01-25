import os
import matplotlib.pyplot as plt
from actions_dict import actions_dict
import numpy as np
from actions_dict import actions_dict
import random
# from Q_Net import DQN, DQN_CNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import deque
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
random goal, random initial position
'''


class Environment(gym.Env):
    def __init__(self, size, init_map, agents_position, mode='eval', save_dir=''):
        self.agent_1 = None
        self.mode = mode
        self.save_dir = save_dir
        self.map_size = size
        self.init_map = init_map.copy()
        self.map = init_map.copy()
        self.agents_position = agents_position
        self.agents_num = 1
        self.init_agents(self.agents_position)
        self.goals = -1 * np.ones([1, 2])

    def init_agents(self, agents_position):
        '''
        初始化agents
        :param agents_position: agents的初始位置
        :return:
        '''
        self.agent_1 = Agent(agents_position[0, :], self.map, agents_position)

        if self.mode == 'eval':
            self.agent_1.load_model(self.save_dir)

        if self.mode == 'train':
            pass

    def reset_env_random(self):
        '''
        重设agent的位置
        :return:
        '''
        feasible_points = np.argwhere(self.init_map == 1)
        # 随机选择三个不同的点作为agent的起始位置
        self.goals = feasible_points[np.random.choice(feasible_points.shape[0], 1, replace=False)]
        # self.goals = feasible_points[[1]]
        # 随机选择三个不同的点作为goal的位置
        # 确保goal的位置与agent的起始位置不同
        self.agents_position = feasible_points[np.random.choice(feasible_points.shape[0], 1, replace=False)]
        while set(map(tuple, self.agents_position)) & set(map(tuple, self.goals)):
            self.agents_position = feasible_points[np.random.choice(feasible_points.shape[0], 1, replace=False)]

        self.map_update(self.agents_position)

        self.agent_1.location_update(self.agents_position[0, :])
        self.agent_1.set_goal(self.goals[0, :])

    def get_state(self, idx):
        '''
        获得特定agent的state
        :param idx: agent的idx
        :return:
        '''
        state = np.concatenate((self.agents_position[idx, :].flatten(), self.map.flatten(),
                                self.agents_position.flatten(), self.goals.flatten()))
        return state

    def take_actions(self, fov, h):
        '''
        获得每个agent的action，并且计算新的agents_position，输入agent_position_update()进行地图跟新
        '''
        action = self.agent_1.act(fov, h)
        self.agent_1.return_action()
        action1_str = [key for key, value in actions_dict.items() if value == action]  # 查找当前数所代表的动作

        self.is_moveable(np.array([action]))  # 判断下一步是否合理
        self.map_update(self.agents_position)  # 更新map的信息

        return action

    def step(self, action):
        '''
        RL训练中，根据state-action pair，获得对应的reward
        :param action:
        :return: state, reward, done
        '''
        image_tensor = torch.tensor(self.map, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)  # [1, 30, 30]
        image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 30, 30]
        transform = transforms.Normalize(mean=[0.5], std=[0.5])
        fov_next_tensor = transform(image_tensor).to(device)
        h_next = np.concatenate((self.agents_position[0, :].flatten(), self.goals.flatten())).copy()
        h_next = torch.FloatTensor(h_next).to(device)
        observation = (fov_next_tensor, h_next)

        reward = -1
        done = False

        mh = np.linalg.norm(self.agents_position[0, :] - self.goals[0, :], ord=1)
        if mh < self.agent_1.dis_to_goal:
            reward = reward + 0.5

        if self.is_reach_goal():
            reward = 10
            done = True

        return observation, reward, done

    def observe(self):
        '''
        观察环境，agent获得map信息、所有agent的路径信息、目标信息
        :return:
        '''
        self.agent_1.observe(self.map, self.agents_position, self.goals)

    def set_goals(self, agent_idx, goal):
        '''设定特定agent的goal坐标'''
        self.goals[agent_idx - 1, :] = goal
        self.agent_1.set_goal(goal)

    def is_reach_goal(self):
        '''
        检测是否有机器人到达goal
        :return:
        '''
        for idx in range(self.agents_num):
            if (self.agents_position[idx, 0] == self.goals[idx, 0]
                    and self.agents_position[idx, 1] == self.goals[idx, 1]):
                print("agent " + str(idx) + " reach goal!")
                return True
        return False

    def is_moveable(self, actions):
        '''
        判断action在map中是否合法
        :param actions: actions所对应的index, 同时输入三个agents的
        :return: 如果action合理，则更新position;否则保持原地不动.返回值为agent position
        '''
        position_ = self.agents_position.copy()
        for i in range(self.agents_num):
            if actions[i] == 3:
                position_[i, 1] += 1
            if actions[i] == 2:
                position_[i, 1] += -1
            if actions[i] == 0:
                position_[i, 0] += -1
            if actions[i] == 1:
                position_[i, 0] += 1

        # 判断下一步是否合理
        for i in range(self.agents_num):
            if self.map[position_[i, 0], position_[i, 1]] != 0:
                self.agents_position[i, :] = position_[i, :]
            else:
                pass

        return self.agents_position

    def map_update(self, agents_position):
        '''
        更新map中的位置信息
        :param agents_position:
        :return:
        '''
        new_map = self.init_map.copy()  # update the map with initial map

        for idx in range(self.agents_num):
            if self.goals[idx, 0] == -1:
                pass
            else:
                new_map[int(self.goals[idx, 0]), int(self.goals[idx, 1])] = 3
        new_map[agents_position[:, 0], agents_position[:, 1]] = 2

        self.map = new_map

    def plot(self):
        '''
        根据map上记录的信息进行绘制
        :return:
        '''
        self.map.astype(np.int8)
        self.pic = plt.imshow(self.map, cmap='Blues', interpolation='none')

        # for i in range(len(self.map)):
        #     for j in range(len(self.map[i])):
        #         plt.annotate(str(round(self.map[i][j])), xy=(j, i), ha='center', va='center')

        plt.grid(which='major', color='black', linestyle='-', linewidth=1)
        plt.xticks(np.arange(-0.5, len(self.map), 1), [])
        plt.yticks(np.arange(-0.5, len(self.map[0]), 1), [])
        plt.axis('off')
        plt.pause(0.05)


class Agent():
    def __init__(self, location, map_observation, agents_observation, gamma=0.95, epsilon=0.9, epsilon_min=0.01, epsilon_decay=0.9, target_update=16):
        self.location = location
        self.action = None
        self.map_observation = map_observation
        self.agents_observation = agents_observation
        self.goal = np.array([-1, -1])
        self.goals = None
        self.trajectory = None
        self.current_step = None

        # dqn setup
        self.state_size = len(self.location.flatten()) + len(self.map_observation.flatten()) + len(self.goal.flatten())
        # self.state_size = len(self.location.flatten()) + len(self.goal.flatten())
        self.action_size = 5
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数

        cnn_output_size = 32  # 假设卷积层输出的特征大小
        vector_size = len(self.location.flatten()) + len(self.goal.flatten())  # 输入向量的特征大小
        hidden_size = 128  # 全连接层的隐藏层大小
        lstm_hidden_size = 32  # LSTM的隐藏状态大小
        lstm_layers = 1  # LSTM的层数

        self.model = CustomNet(cnn_output_size, vector_size, hidden_size, lstm_hidden_size, lstm_layers).to(device)
        self.target_model = CustomNet(cnn_output_size, vector_size, hidden_size, lstm_hidden_size, lstm_layers).to(device)

        # self.model = DQN_CNN().to(device)
        # self.target_model = DQN_CNN().to(device)

        self.dis_to_goal = 1000
        self.loss = 100
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def observe(self, map_observation, agents_observation, goals):
        '''获得观察'''
        self.map_observation = map_observation
        self.agents_observation = agents_observation
        self.goals = goals

    def location_update(self, location):
        '''更新位置'''
        self.location = location

    def set_goal(self, goal_position):
        '''设定目标点'''
        self.goal = goal_position

    def reset(self):
        '''暂停任务'''
        self.goal = None

    def set_trajectory(self, trajectory):
        '''设定轨迹'''
        self.trajectory = trajectory
        self.current_step = 1  # 从轨迹的第二个点开始

    def take_action_tracking(self):
        '''根据轨迹执行'''
        if self.trajectory is None:
            print("No trajectory for agent")
            return None
        else:
            if self.current_step is None:
                print("current step is None")
                return None
            else:
                self.action = self.trajectory[self.current_step]

        return self.action

    def take_action_keybroad(self, action):
        '''根据键盘输入执行'''
        self.action = action
        return self.action

    def take_action_random(self):
        '''随机生成actions'''
        random_int = random.randint(0, 4)
        self.action = random_int
        return self.action

    def return_action(self):
        '''
        用于查询action
        :return: 返回当前agent的action
        '''
        action = self.action
        self.action = 4
        if action is None:
            action = self.action
        return action

    def remember(self, fov, h, action, reward, fov_next, h_next, done):
        '''
        加入Replay Buffer中
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        '''
        self.memory.append((fov, h, action, reward, fov_next, h_next, done))

    def take_action_rl(self, fov, h):
        '''
        根据DQN网络生成所需要执行的action
        :param state:
        :return: action
        '''
        # state = torch.FloatTensor(state).to(device)
        act_values = self.model(fov, h)
        self.action = np.argmax(act_values.cpu().detach().numpy())
        return np.argmax(act_values.cpu().detach().numpy())

    def act(self, fov, h):
        '''
        根据DQN网络生成所需要执行的action
        :param state:
        :return: action
        '''
        if np.random.rand() <= self.epsilon:
            self.action = random.randrange(self.action_size)
            return random.randrange(self.action_size)

        act_values = self.model(fov, h)
        self.action = np.argmax(act_values.cpu().detach().numpy())

        return np.argmax(act_values.cpu().detach().numpy())

    def load_model(self, path):
        '''
        load the training model
        :param path: model path
        :return:
        '''
        self.model = torch.load(path)
        # self.model.load_state_dict(torch.load(path))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        loss_mean = 0

        for fov, h, action, reward, fov_next, h_next, done in minibatch:
            action = torch.LongTensor([action]).cuda()  # Move to GPU
            reward = torch.FloatTensor([reward]).cuda()  # Move to GPU
            done = torch.FloatTensor([done]).cuda()  # Move to GPU

            with torch.no_grad():
                target = reward
                if not done:
                    # 选取下一状态的最大Q值
                    target_next = self.target_model(fov_next, h_next).max(1)[0]
                    target = reward + (self.gamma * target_next)

            # 获取预测的Q值
            current_q_values = self.model(fov, h).gather(1, action.unsqueeze(1)).squeeze(1)

            # 计算当前Q值和目标Q值之间的均方误差损失
            loss = nn.MSELoss()(current_q_values, target)

            loss_mean += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # print("loss: ", loss_mean/batch_size)
        self.loss = loss_mean/batch_size

        if self.count % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())  # 更新目标网络

        self.count += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class CustomNet(nn.Module):
    def __init__(self, cnn_output_size, vector_size, hidden_size, lstm_hidden_size, lstm_layers, output_size=5):
        super(CustomNet, self).__init__()
        # CNN部分
        # 假设输入图像是单通道的，如果是多通道需要更改in_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)

        # 池化操作将图像大小减半，所以两次卷积和池化后，图像的大小变为 30/2/2 = 7（向下取整）
        self.fc1 = nn.Linear(20 * 7 * 7, 100)
        self.fc_for_cnn = nn.Linear(100, cnn_output_size)

        # 单独的输入向量处理部分
        self.fc_for_vector = nn.Linear(vector_size, hidden_size)
        # 将CNN输出和输入向量的特征相加后的全连接层
        self.fc_combined = nn.Linear(cnn_output_size + hidden_size, hidden_size)
        # LSTM部分
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers)
        # 动作向量输出部分
        self.fc_action = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x_cnn, x_vector):
        # CNN部分
        # 应用第一个卷积层 + ReLU激活函数 + 池化
        x_cnn = self.pool(F.relu(self.conv1(x_cnn)))
        # 应用第二个卷积层 + ReLU激活函数 + 池化
        x_cnn = self.pool(F.relu(self.conv2(x_cnn)))
        # 展平特征图
        x_cnn = x_cnn.view(-1, 20 * 7 * 7)
        # 应用第一个全连接层 + ReLU激活函数
        x_cnn = F.relu(self.fc1(x_cnn))
        # 应用第二个全连接层（输出层）
        x_cnn = self.fc_for_cnn(x_cnn)
        # 独立输入向量部分
        x_vector = self.fc_for_vector(x_vector)
        x_vector = x_vector.unsqueeze(0)
        # 合并CNN输出和输入向量
        combined = torch.cat((x_cnn, x_vector), dim=1)
        # 经过全连接层
        combined_fc = self.fc_combined(combined)
        # ReLU激活
        activated = F.relu(combined_fc)
        # LSTM部分（假设只有一个序列步长）
        activated = activated.unsqueeze(0)  # 增加一个序列长度的维度
        lstm_out, _ = self.lstm(activated)
        # 取 LSTM 的最后一个时间步的输出
        lstm_out = lstm_out[-1]
        # 生成动作向量
        action = self.fc_action(lstm_out)
        return action


def get_next_available_dir(base_dir):
    dir_number = 0
    while True:
        dir_number += 1
        new_dir = f"{base_dir}_{dir_number}" if dir_number > 1 else base_dir
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        if not os.listdir(new_dir):  # Check if the directory is empty
            return new_dir


def save_model(agent, episode, save_dir='./train'):
    # 确保使用唯一的目录

    # 生成保存模型的路径，包含episode序号
    model_path = os.path.join(save_dir, f'model_episode_{episode}.pt')

    # 保存模型
    torch.save(agent.model, model_path)
    print(f'Model saved to {model_path}.')


def eval_dqn(filename):
    size = 30
    init_map = np.zeros([size, size])
    init_map[1, 1:size - 1] = 1
    init_map[1:size - 1, 1] = 1
    init_map[size - 2, 2:size - 2] = 1
    init_map[2:size - 1, size - 2] = 1
    init_map[2:size - 2, 10] = 1
    init_map[2:size - 2, 20] = 1
    init_map[22, 2:size - 2] = 1
    init_map[8, 2:size - 2] = 1

    agents_pos = np.array([[1, 2]])
    env = Environment(size, init_map, agents_pos, mode='eval', save_dir=filename)
    env.init_map = init_map
    env.reset_env_random()
    # env.set_goals(1, [5, 1])
    state = env.get_state(idx=0)

    cnt_step = 0
    while True:
        cnt_step += 1
        fov = env.map
        h = np.concatenate((env.agents_position[0, :].flatten(), env.goals.flatten())).copy()

        image_tensor = torch.tensor(fov, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)  # [1, 30, 30]
        image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 30, 30]
        transform = transforms.Normalize(mean=[0.5], std=[0.5])
        # 应用这个变换
        image_tensor = transform(image_tensor)
        fov_tensor = image_tensor.to(device)
        h_tensor = torch.FloatTensor(h).to(device)

        env.plot()  # 绘制map
        env.agent_1.take_action_rl(fov_tensor, h_tensor)
        action = env.take_actions()
        next_state, reward, done = env.step(action)

        if done or cnt_step > 40:
            cnt_step = 0
            env.reset_env_random()
        env.plot()


def train_dqn(episode_count, batch_size, max_step, continue_train=False, model_file=""):
    size = 30
    init_map = np.zeros([size, size])
    init_map[1, 1:size - 1] = 1
    init_map[1:size - 1, 1] = 1
    init_map[size - 2, 2:size - 2] = 1
    init_map[2:size - 1, size - 2] = 1
    init_map[2:size - 2, 10] = 1
    init_map[2:size - 2, 20] = 1
    init_map[22, 2:size - 2] = 1
    init_map[8, 2:size - 2] = 1

    loss_crv = np.array([])

    agents_pos = np.array([[1, 2]])

    env = Environment(size, init_map, agents_pos, mode='train')
    env.init_map = init_map

    save_dir = get_next_available_dir('./train_test')

    for episode in range(episode_count):
        print("episode:", episode)
        env.reset_env_random()
        # state = env.get_state(idx=0)
        # state_size = len(state)
        done = False

        t = 0
        while not done and t <= max_step:
            t += 1
            fov = env.map
            h = np.concatenate((env.agents_position[0, :].flatten(), env.goals.flatten())).copy()

            image_tensor = torch.tensor(fov, dtype=torch.float32)
            image_tensor = image_tensor.unsqueeze(0)  # [1, 30, 30]
            image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 30, 30]
            transform = transforms.Normalize(mean=[0.5], std=[0.5])
            # 应用这个变换
            image_tensor = transform(image_tensor)
            fov_tensor = image_tensor.to(device)
            h = torch.FloatTensor(h).to(device)

            action = env.take_actions(fov_tensor, h)  # 更新position，map等环境信息，在step函数中无需重复进行

            next_state, reward, done = env.step(action)

            image_tensor = torch.tensor(env.map, dtype=torch.float32)
            image_tensor = image_tensor.unsqueeze(0)  # [1, 30, 30]
            image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 30, 30]
            transform = transforms.Normalize(mean=[0.5], std=[0.5])
            fov_next_tensor = transform(image_tensor).to(device)
            # h_ = torch.FloatTensor(h).to(device)
            h_next = np.concatenate((env.agents_position[0, :].flatten(), env.goals.flatten())).copy()
            h_next = torch.FloatTensor(h_next).to(device)
            env.agent_1.remember(fov_tensor, h, action, reward, fov_next_tensor, h_next, done)

            if len(env.agent_1.memory) > batch_size:
                # print(len(env.agent_1.memory))
                env.agent_1.replay(batch_size)

        print("loss: ", env.agent_1.loss)
        loss_crv = np.append(loss_crv, env.agent_1.loss)  # 记录loss曲线

        if (episode % 500) == 0:
            save_model(env.agent_1, episode, save_dir=save_dir)
            np.save(save_dir + "/loss.npy", loss_crv)


if __name__ == '__main__':
    train_dqn(100000, 16, max_step=50)
    # eval_dqn('train_cnn_3/model_episode_19000.pt')
