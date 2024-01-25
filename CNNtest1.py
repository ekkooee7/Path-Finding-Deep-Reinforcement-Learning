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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
fixed goal
'''

class Environment():
    def __init__(self, size, init_map, agents_position, mode='eval', save_dir='train_10/model_episode_18000.pt'):
        self.agent_1 = None
        self.mode = mode
        self.save_dir = save_dir
        self.map_size = size
        self.init_map = init_map.copy()
        self.map = init_map.copy()
        self.agents_position = agents_position
        self.agents_num = 1
        # self.fig = plt.subplots()
        # self.agent_1 = Agent(self.agents_position[0, :], self.map, self.agents_position)
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
        # self.goals = feasible_points[np.random.choice(feasible_points.shape[0], 1, replace=False)]
        self.goals = feasible_points[[1]]
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

    def take_actions(self):
        '''
        获得每个agent的action，并且计算新的agents_position，输入agent_position_update()进行地图跟新
        '''
        action1 = self.agent_1.return_action()
        action1_str = [key for key, value in actions_dict.items() if value == action1]  # 查找当前数所代表的动作

        self.is_moveable(np.array([action1]))  # 判断下一步是否合理
        # done = self.is_reach_goal()
        # self.agent_1
        self.map_update(self.agents_position)  # 更新map的信息

    def step(self, idx):
        '''
        RL训练中，根据state-action pair，获得对应的reward
        :param action:
        :return: state, reward, done
        '''
        state = np.concatenate((self.agents_position[idx, :].flatten(), self.map.flatten(),
                                self.agents_position.flatten(), self.goals.flatten()))
        # reward = np.linalg.norm(self.goals[idx, :] - self.agents_position[idx, :]) / 1000
        reward = -1
        done = False

        mh = np.linalg.norm(self.agents_position[idx, :] - self.goals[idx, :], ord=1)
        if mh < self.agent_1.dis_to_goal:
            reward = reward + 0.5

        if self.is_reach_goal():
            reward = 10
            done = True

        return state, reward, done

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

        self.model = ActionCNN().to(device)
        self.target_model = ActionCNN().to(device)

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

    def remember(self, state, action, reward, next_state, done):
        '''
        加入Replay Buffer中
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        '''
        self.memory.append((state, action, reward, next_state, done))

    def take_action_rl(self, state):
        '''
        根据DQN网络生成所需要执行的action
        :param state:
        :return: action
        '''
        # state = torch.FloatTensor(state).to(device)
        act_values = self.model(state)
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
        # fov = torch.FloatTensor(fov).to(device)
        act_values = self.model(fov)
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


    def replay1(self, batch_size):
        '''
        从replay buffer中采样动作，进行value update
        :param batch_size:
        :return:
        '''
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model(next_state).detach().cpu().numpy()))
            target_f = self.model(state)

            # print(target_f)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        loss_mean = 0

        for state, action, reward, next_state, done in minibatch:
            # state = torch.FloatTensor(state).unsqueeze(0).cuda()  # Move to GPU
            # next_state = torch.FloatTensor(next_state).unsqueeze(0).cuda()  # Move to GPU
            action = torch.LongTensor([action]).cuda()  # Move to GPU
            reward = torch.FloatTensor([reward]).cuda()  # Move to GPU
            done = torch.FloatTensor([done]).cuda()  # Move to GPU

            with torch.no_grad():
                target = reward
                if not done:
                    # 选取下一状态的最大Q值
                    target_next = self.target_model(next_state).max(1)[0]
                    target = reward + (self.gamma * target_next)

            # 获取预测的Q值
            current_q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)

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


    def replay11(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        next_states = torch.tensor(next_states, dtype=torch.float).cuda()
        states = torch.tensor(states, dtype=torch.float).cuda()

        model_predict = self.model(next_states)
        # print(model_predict.shape)
        model_predict_max = model_predict.max(1)[0].view(-1, 1).cpu().detach().numpy()
        model_predict_max = np.squeeze(model_predict_max)

        targets = rewards + self.gamma * model_predict_max * (1-dones)
        targets = torch.tensor(targets, dtype=torch.float).view(-1, 1).cuda()

        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1).cuda()
        q_value = self.model(states).view(-1, 2)
        q_value = q_value.gather(1, actions)

        self.optimizer.zero_grad()

        mse = F.mse_loss(q_value, targets)
        loss = torch.mean(mse)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        states = states.cpu().detach().numpy()
        next_states = next_states.cpu().detach().numpy()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class ActionCNN(nn.Module):
    def __init__(self):
        super(ActionCNN, self).__init__()

        # 假设输入图像是单通道的，如果是多通道需要更改in_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)

        # 池化操作将图像大小减半，所以两次卷积和池化后，图像的大小变为 30/2/2 = 7（向下取整）
        self.fc1 = nn.Linear(20 * 7 * 7, 100)
        self.fc2 = nn.Linear(100, 5)

    def forward(self, x):
        # 应用第一个卷积层 + ReLU激活函数 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 应用第二个卷积层 + ReLU激活函数 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 20 * 7 * 7)
        # 应用第一个全连接层 + ReLU激活函数
        x = F.relu(self.fc1(x))
        # 应用第二个全连接层（输出层）
        x = self.fc2(x)
        return x


class ActionCNNWithConcat(nn.Module):
    def __init__(self, additional_vector_length):
        super(ActionCNNWithConcat, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)

        # 全连接层的输入需要加上额外向量的长度
        self.fc1_input_size = 20 * 7 * 7 + additional_vector_length
        self.fc1 = nn.Linear(self.fc1_input_size, 100)
        self.fc2 = nn.Linear(100, 5)

    def forward(self, x, additional_vector):
        # 应用卷积层和池化层
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 20 * 7 * 7)
        # 将展平后的特征图与额外的向量拼接
        x = torch.cat((x, additional_vector), dim=1)
        # 应用全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
        # 将 NumPy 数组转换为 PyTorch 张量
        image_tensor = torch.tensor(fov, dtype=torch.float32)
        # 添加一个通道维度，使其成为 [C, H, W] 的形状，对于灰度图像，C=1
        image_tensor = image_tensor.unsqueeze(0)  # [1, 30, 30]
        # 添加一个批量维度，使其成为 [B, C, H, W] 的形状，这里 B=1
        image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 30, 30]
        # 如果你的模型要求输入数据在 0 到 1 之间，确保你的数据被正确归一化
        # 如果数据已经在这个范围内，你可以跳过这个步骤
        # 如果需要进行其他预处理，比如数据归一化，可以使用 torchvision.transforms
        # 例如，你可以定义一个 transform 来进行归一化
        transform = transforms.Normalize(mean=[0.5], std=[0.5])
        # 应用这个变换
        image_tensor = transform(image_tensor)
        fov_tensor = image_tensor.to(device)
        state = fov_tensor

        env.plot()  # 绘制map
        env.agent_1.take_action_rl(state)
        env.take_actions()
        next_state, reward, done = env.step(idx=0)

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

    save_dir = get_next_available_dir('./train_cnn')

    for episode in range(episode_count):
        # agent = Agent(env.agents_position[0, :], env.map, env.agents_position, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9, target_update=10)
        print("episode:", episode)
        env.reset_env_random()
        # state = env.get_state(idx=0)
        # state_size = len(state)
        done = False

        t = 0
        while not done and t <= max_step:
            t += 1
            state = np.concatenate((env.agents_position[0, :].flatten(), env.map.flatten(), env.goals.flatten()))
            # state = env.map
            # state = np.concatenate((env.agents_position[0, :].flatten(), env.goals.flatten()))
            fov = env.map
            # 将 NumPy 数组转换为 PyTorch 张量
            image_tensor = torch.tensor(fov, dtype=torch.float32)
            # 添加一个通道维度，使其成为 [C, H, W] 的形状，对于灰度图像，C=1
            image_tensor = image_tensor.unsqueeze(0)  # [1, 30, 30]
            # 添加一个批量维度，使其成为 [B, C, H, W] 的形状，这里 B=1
            image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 30, 30]
            # 如果你的模型要求输入数据在 0 到 1 之间，确保你的数据被正确归一化
            # 如果数据已经在这个范围内，你可以跳过这个步骤
            # 如果需要进行其他预处理，比如数据归一化，可以使用 torchvision.transforms
            # 例如，你可以定义一个 transform 来进行归一化
            transform = transforms.Normalize(mean=[0.5], std=[0.5])
            # 应用这个变换
            image_tensor = transform(image_tensor)
            fov_tensor = image_tensor.to(device)
            state = fov_tensor

            h = np.concatenate((env.agents_position[0, :].flatten(), env.goals.flatten()))
            action = env.agent_1.act(fov_tensor, h)
            env.take_actions()  # 更新position，map等环境信息，在step函数中无需重复进行
            next_state, reward, done = env.step(idx=0)
            # next_state = np.concatenate((env.agents_position[0, :].flatten(), env.map.flatten(), env.goals.flatten()))
            # next_state = np.concatenate((env.agents_position[0, :].flatten(), env.goals.flatten()))
            image_tensor = torch.tensor(env.map, dtype=torch.float32)
            image_tensor = image_tensor.unsqueeze(0)  # [1, 30, 30]
            image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 30, 30]
            transform = transforms.Normalize(mean=[0.5], std=[0.5])
            image_tensor = transform(image_tensor)
            next_state = image_tensor.to(device)

            env.agent_1.remember(state, action, reward, next_state, done)

            if len(env.agent_1.memory) > batch_size:
                # print(len(env.agent_1.memory))
                env.agent_1.replay(batch_size)

        print("loss: ", env.agent_1.loss)
        loss_crv = np.append(loss_crv, env.agent_1.loss)  # 记录loss曲线

        if (episode % 500) == 0:
            save_model(env.agent_1, episode, save_dir=save_dir)
            np.save(save_dir + "/loss.npy", loss_crv)


if __name__ == '__main__':
    # train_dqn(100000, 16, max_step=40)
    eval_dqn('train_cnn/model_episode_2000.pt')
