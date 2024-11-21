import yaml
import gym
from gym import spaces
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from collections import deque
from tqdm import tqdm
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

config = yaml.safe_load(open("config/user.yaml", 'r', encoding='utf-8'))
initial_cash = config['user']['initial_cash']

# 设定PyTorch使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StockTradingEnv(gym.Env):
    def __init__(self, data,buy_ratio=0.5,sell_ratio=0.5):
        super(StockTradingEnv, self).__init__()
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        self.data = data
        self.current_step = 0
        self.done = False

        # 动作空间：0 - 持有，1 - 买入，2 - 卖出
        self.action_space = spaces.Discrete(3)

        # 状态空间：当前持有股票数量、当前现金余额、当前股票价格、买入比例、卖出比例、移动平均线、RSI等
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)

        # 初始状态
        self.initial_balance = initial_cash
        self.transaction_fee_rate = 0.0005  # 手续费万分之五
        self.reset()

        # 用于记录每日数据
        self.history = []
        self.daily_operations = []  # 每日操作日志

        # 技术指标参数
        self.window_size = 14  # RSI窗口大小
        self.prices = []  # 存储历史价格用于计算指标

    def reset(self):
        self.current_step = 0
        self.done = False
        self.current_balance = self.initial_balance
        self.current_stock_owned = 0
        self.current_stock_price = self.data[self.current_step]['close']

        self.history = []  # 清空历史记录
        self.daily_operations = []  # 清空每日操作日志
        self.prices = []  # 清空价格历史

        # 随机生成买入和卖出比例

        return torch.tensor([self.current_stock_owned, self.current_balance, self.current_stock_price, self.buy_ratio, self.sell_ratio, 0, 0], dtype=torch.float32, device=device)
    
    def calculate_reward(self, current_balance, previous_balance, trade_flag):
        rate_of_return = (current_balance - previous_balance) / previous_balance
        reward = rate_of_return
        # 频繁交易奖励
        if trade_flag:
            reward += 0.1
        # 如果为负数，惩罚增加
        if reward < 0:
            reward *= 2
        return reward
    
    def calculate_indicators(self):
        # 计算简单移动平均线和相对强弱指数（RSI）
        if len(self.prices) >= 2:
            sma = np.mean(self.prices[-2:])  # 最近两个价格的简单移动平均
        else:
            sma = 0

        if len(self.prices) >= self.window_size:
            delta = np.diff(self.prices[-self.window_size:], n=1)
            gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
            loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
            rs = gain / loss if loss != 0 else 0
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 0

        return sma, rsi

    # 在环境类中
    def step(self, action):
        buy_today = False  # 今天是否买入
        sell_today = False
        previous_balance = self.current_balance
        self.prices.append(self.current_stock_price)  # 记录当前价格用于计算指标
        sma, rsi = self.calculate_indicators()  # 计算指标

        # 执行动作
        if action == 1:  # 买入
            max_buy_amount = int(self.current_balance // self.current_stock_price)
            buy_amount = int(max_buy_amount * self.buy_ratio)
            # 买入只能是1手（100股）的整数倍
            buy_amount = buy_amount - buy_amount % 100

            if buy_amount > 0:
                buy_today = True # 今天买入
                # 计算手续费
                fee = buy_amount * self.current_stock_price * self.transaction_fee_rate
                self.current_stock_owned += buy_amount
                self.current_balance -= buy_amount * self.current_stock_price + fee
                # 记录每日买入操作
                self.daily_operations.append({
                    'time': self.data[self.current_step]['date'].strftime('%Y-%m-%d'),
                    'action': 'buy',
                    'price': self.current_stock_price,
                    'amount': buy_amount,
                    'fee': fee,
                    'cash': self.current_balance,
                    'total_value': self.current_balance + self.current_stock_owned * self.current_stock_price,
                    'sma': sma,  # 添加简单移动平均线
                    'rsi': rsi,
                    'stock_owned': self.current_stock_owned, # 添加持有股票数量
                    'stock_value': self.current_stock_owned * self.current_stock_price # 添加持有股票市值
                })

        elif action == 2:  # 卖出
            max_sell_amount = self.current_stock_owned
            sell_amount = int(max_sell_amount * self.sell_ratio)
            # 卖出只能是1手（100股）的整数倍
            sell_amount = sell_amount - sell_amount % 100

            if sell_amount > 0:
                # 今天卖出
                sell_today = True
                # 计算手续费
                fee = sell_amount * self.current_stock_price * self.transaction_fee_rate
                self.current_stock_owned -= sell_amount
                self.current_balance += sell_amount * self.current_stock_price - fee
                # 记录每日卖出操作
                self.daily_operations.append({
                    'time': self.data[self.current_step]['date'].strftime('%Y-%m-%d'),
                    'action': 'sell',
                    'price': self.current_stock_price,
                    'amount': sell_amount,
                    'fee': fee,
                    'cash': self.current_balance,
                    'total_value': self.current_balance + self.current_stock_owned * self.current_stock_price,
                    'sma': sma,  # 添加简单移动平均线
                    'rsi': rsi,
                    'stock_owned': self.current_stock_owned, # 添加持有股票数量
                    'stock_value': self.current_stock_owned * self.current_stock_price # 添加持有股票市值
                })
        else: # 持有
            self.daily_operations.append({
                'time': self.data[self.current_step]['date'].strftime('%Y-%m-%d'),
                'action': 'hold',
                'price': self.current_stock_price,
                'amount': 0,
                'fee': 0,
                'cash': self.current_balance,
                'total_value': self.current_balance + self.current_stock_owned * self.current_stock_price,
                'sma': sma,  # 添加简单移动平均线
                'rsi': rsi,  # 添加RSI
                'stock_owned': self.current_stock_owned, # 添加持有股票数量
                'stock_value': self.current_stock_owned * self.current_stock_price # 添加持有股票市值
            })

        # 更新股票价格
        self.current_step += 1
        if self.current_step < len(self.data):
            self.current_stock_price = self.data[self.current_step]['close']
        else:
            self.done = True  # 数据结束，标记为完成
            
        # 定义是否交易的标志
        trade_flag = buy_today or sell_today

        # 计算奖励
        reward = self.calculate_reward(self.current_balance, previous_balance,trade_flag)

        # 记录每日数据
        total_value = self.current_balance + self.current_stock_owned * self.current_stock_price
        self.history.append({
            'total_value': total_value,
            'cash': self.current_balance,
            'stock_value': self.current_stock_owned * self.current_stock_price,
            'step': self.current_step,
            'time': self.data[self.current_step]['date'] if self.current_step < len(self.data) else None,  # 添加时间
            'buy_ratio': self.buy_ratio,  # 买入比例
            'sell_ratio': self.sell_ratio,  # 卖出比例
            'sma': sma,  # 添加简单移动平均线
            'rsi': rsi  # 添加RSI
        })
        

        # 返回状态、奖励、是否结束、额外信息
        state = torch.tensor([self.current_stock_owned, self.current_balance, self.current_stock_price, self.buy_ratio, self.sell_ratio, sma, rsi], dtype=torch.float32).to(device)
        return state, reward, self.done, {}



    def render(self, mode='human'):
        pass  # 不输出每一步的操作
    
epsilon_max_global = 0.5  # 全局最大探索率
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # 经验回放缓冲区
        self.gamma = 0.99  # 折扣因子
        self.epsilon = epsilon_max_global  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.learning_rate = 0.0005  # 学习率
        self.model = self._build_model()  # 创建神经网络模型
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 定义优化器
        self.criterion = nn.MSELoss()  # 定义损失函数
        
        self.buy_today = False  # 今天是否买入
        self.sell_today = False  # 今天是否卖出

    def _build_model(self):
        class QNetwork(nn.Module):
            def __init__(self, state_size, action_size, dropout_rate=0.2):
                super(QNetwork, self).__init__()
                # 定义隐藏层
                self.fc1 = nn.Linear(state_size, 128)  
                self.fc2 = nn.Linear(128, 256)
                #self.gru = nn.GRU(256, 256, 2, batch_first=True, dropout=dropout_rate)
                self.fc3 = nn.Linear(256, action_size)
                self.dropout = nn.Dropout(dropout_rate)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                # x, _ = self.gru(x.unsqueeze(0))
                
                x = self.fc3(x)
                return x
            
        model = QNetwork(self.state_size, self.action_size).to(device)
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = state.clone().detach().requires_grad_(True).to(device)
            next_state = next_state.clone().detach().requires_grad_(True).to(device)

            target = reward
            if not done:
                # 使用双重DQN
                target += self.gamma * torch.max(self.model(next_state).detach())

            target_f = self.model(state).detach().cpu().numpy().squeeze()
            if len(target_f.shape) == 1:  
                target_f = target_f.reshape(1, -1)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), torch.FloatTensor(target_f).to(device))
            loss.backward()
            self.optimizer.step()
            
    # 定义保存和加载模型的方法
    def save_model(self, episode):
        torch.save(self.model.state_dict(), f'dqn_model_{episode}.pth')

    def load_model(self, episode):
        self.model.load_state_dict(torch.load(f'dqn_model_{episode}.pth', weights_only=True))


def train_agent(env, agent, episodes=1000, batch_size=32, start_episode=0, base_return=0):
    previous_profit = 0
    min_profit = 10
    max_profit = 0
    all_daily_operations = []  # 存储所有 daily_operations
    initial_profit = base_return * 100 # 用区间涨幅作为初始目标收益率
    target_profit = initial_profit  # 目标收益率

    for e in range(episodes):
        state = env.reset()
        state = state.clone().detach().requires_grad_(True).to(device)
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.clone().detach().requires_grad_(True).to(device)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
        # 每次训练结束后记录 daily_operations
        all_daily_operations.append(env.daily_operations.copy())
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        # 更新探索率和目标收益率
        profit = calculate_profit(all_daily_operations[-1])

        # 1. 动态调整目标收益率和探索率
        if profit > target_profit:
            # 如果收益率超过目标收益率，提高目标收益率
            target_profit = min(profit, target_profit + 1)  # 提高目标收益率 1%
            print(f"[REWARD] 第{e + 1 + start_episode}次训练，收益率{profit:.2f}%超过了目标收益率: {target_profit:.2f}%", f"目标收益率提高到: {target_profit:.2f}%", f"当前最高收益率: {max_profit:.2f}%")
            agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)  # 衰减探索率
            # 如果收益率超过最大收益率，更新最大收益率，并且超过目标收益率0.5%以上，奖励探索率衰减
            if profit > max_profit:
                # 如果max_profit为0
                if max_profit == 0:
                    if profit > 0:
                        max_profit = profit
                        print(f"[INIT] 第{e + 1 + start_episode}次训练，初始化最大收益率为: {max_profit:.2f}%")
                else:
                    over_profit = profit - target_profit
                    over_profit_rate = over_profit / target_profit
                    reward_num = over_profit_rate // 0.4
                    decay_rate = 0.99 ** reward_num * agent.epsilon_decay
                    agent.epsilon = max(agent.epsilon * decay_rate, agent.epsilon_min)
                    max_profit = profit
                    print(f"[REWARD] 第{e + 1 + start_episode}次训练，收益率{profit:.2f}%超过了历史最高收益率！探索率衰减比率: {decay_rate:.4f}", f"当前最高收益率: {max_profit:.2f}%")
                
        elif profit < min_profit:
            # 如果收益率为负数，降低目标收益率
            target_profit = max(initial_profit, target_profit - 0.1)  # 降低目标收益率 0.1%
            min_profit = profit
            print(f"[MIN] 第{e + 1 + start_episode}次训练，最低收益率降低到: {min_profit:.2f}%", f"目标收益率降低到: {target_profit:.2f}%", f"当前最低收益率: {min_profit:.2f}%")
            # 略微增加探索率
            agent.epsilon = min(agent.epsilon * 1.01, epsilon_max_global)  # 增加探索率
        
        # 如果收益为正数，减少探索率
        if profit > 0:
            agent.epsilon = max(agent.epsilon * 0.9995, agent.epsilon_min)  # 衰减探索率
        # 如果收益率为负数，增加探索率
        if profit < 0:
            agent.epsilon = min(agent.epsilon * 1.0001, epsilon_max_global)  # 增加探索率
            
        # 记录上次收益率
        previous_profit = profit

        # 保存模型的训练效果数据到csv文件
        profit = calculate_profit(all_daily_operations[-1])
        with open("training_results.csv", "a") as f:
            f.write(f"{e + 1 + start_episode},{profit},{agent.epsilon}\n")

        # 每100次保存一次模型
        if (e + 1 + start_episode) % 100 == 0:
            agent.save_model(e + 1 + start_episode)
    
    return all_daily_operations
        
def print_daily_operations(operations):
    for op in operations:
        # 输出每日操作日志
        date = op['time']
        stock_value = op['amount'] * op['price']
        total_value = op['total_value']
        fee = op['fee']
        amount = op['amount']
        action = op['action']
        percentage = (stock_value / initial_cash) * 100
        # 输出 对齐
        if action == 'buy':
            print(f"日期: {date}, 买入: {amount}股, 价格: {op['price']:.2f}, 费用: {fee:.2f}, 持仓市值: {stock_value:.2f}, 总市值: {total_value:.2f}, 占比: {percentage:.2f}%")
        elif action == 'sell':
            print(f"日期: {date}, 卖出: {amount}股, 价格: {op['price']:.2f}, 费用: {fee:.2f}, 持仓市值: {stock_value:.2f}, 总市值: {total_value:.2f}, 占比: {percentage:.2f}%")
        else:
            print(f"日期: {date}, 无操作")

def calculate_profit(operations):
    # 计算总资产变化的收益率
    if not operations:
        return 0
    initial_value = initial_cash
    final_value = operations[-1]['total_value']  # 最后一次操作的总资产
    return (final_value - initial_value) / initial_value * 100  # 计算收益率（百分比）