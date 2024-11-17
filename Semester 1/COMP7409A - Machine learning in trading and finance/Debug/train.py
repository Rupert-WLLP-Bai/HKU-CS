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
from model.env import DQNAgent, StockTradingEnv, calculate_profit, train_agent
from data.stock_data import get_stock_data, get_stock_name
import threading

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

config = yaml.safe_load(open("config/user.yaml", 'r', encoding='utf-8'))
initial_cash = config['user']['initial_cash']
stock_config = yaml.safe_load(open('config/stock.yaml', 'r', encoding='utf-8'))
start_date = stock_config['start_date']
end_date = stock_config['end_date']
stock_list = stock_config['stock_list']

def train_all(epoch=500):
    for stock in stock_list:
        code = stock['id']
        name = stock['name']
        test_data = get_stock_data(code, start_date, end_date)
        train(test_data, code, name, epoch=epoch)

def train(test_data, id, name,epoch=500):
    df = test_data.copy()
    df.reset_index(inplace=True)
    df['date'] = pd.to_datetime(df['datetime'])
    base_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]

    # 询问是否继续训练
    start_episode = 0
    load_episode = None
    # load_episode = input("请输入要加载的训练次数（若不加载，请直接回车）：")
    if load_episode:
        start_episode = int(load_episode)

    env = StockTradingEnv(df.to_dict(orient='records'))
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # 如果加载训练次数，则加载模型
    if start_episode > 0:
        agent.load_model(start_episode)

    # 训练智能体 并且返回每日操作日志
    episodes = epoch
    print(f"开始训练 {id} {name} {episodes} 次")
    all_daily_operations = train_agent(env, agent, episodes=episodes, batch_size=64, start_episode=start_episode, base_return=base_return)
    

    best_operations = max(all_daily_operations, key=lambda x: calculate_profit(x))
    import matplotlib.dates as mdates

    # 输出最佳利率
    operation_df = pd.DataFrame(best_operations)
    operation_df['time'] = pd.to_datetime(operation_df['time'])
    profit_rate = calculate_profit(best_operations)

    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.figure(figsize=(28, 14))

    # 子图1: 股票收盘价和买卖信号
    plt.subplot(3, 1, 1)
    plt.plot(df['date'], df['close'], label='收盘价')
    plt.scatter(operation_df[operation_df['action'] == 'buy']['time'], operation_df[operation_df['action'] == 'buy']['price'], marker='^', c='g', label='买入')
    plt.scatter(operation_df[operation_df['action'] == 'sell']['time'], operation_df[operation_df['action'] == 'sell']['price'], marker='v', c='r', label='卖出')
    plt.title(f'DQN - {id} {name} Profit Rate: {profit_rate:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()

    # 设置日期格式为按月显示
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # 自动选择合适的日期间隔
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 按 "年-月" 格式显示日期
    plt.xticks(rotation=45)  # 旋转日期标签，避免重叠

    # 子图2: 每日的持仓和总金额
    plt.subplot(3, 1, 2)
    operation_df['total_value'] = operation_df['cash'] + operation_df['stock_value']
    plt.plot(operation_df['time'], operation_df['total_value'], label='总金额')
    plt.plot(operation_df['time'], operation_df['stock_value'], label='持仓市值')
    plt.title('Daily Holdings and Total Value (Best Operations)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()

    # 收益率曲线 低于0的部分标记为绿色 高于0的部分标记为红色 低于10%的部分标记为蓝色 高于10%的部分标记为橙色
    plt.subplot(3, 1, 3)
    operation_df['rate_of_return'] = (operation_df['total_value'] - initial_cash) / initial_cash
    plt.plot(operation_df['time'], operation_df['rate_of_return'], label='Rate of Return')
    # 首先找到上界和下界 作为图标的纵轴
    upper_bound = operation_df['rate_of_return'].max()
    lower_bound = operation_df['rate_of_return'].min()
    plt.fill_between(operation_df['time'], lower_bound, 0, where=(operation_df['rate_of_return'] <= 0), color='green', alpha=0.3, label='Negative Return')
    plt.fill_between(operation_df['time'], 0, upper_bound, where=(operation_df['rate_of_return'] > 0) & (operation_df['rate_of_return'] < 0.1), color='red', alpha=0.3, label='Positive Return')
    plt.fill_between(operation_df['time'], 0.1, upper_bound, where=(operation_df['rate_of_return'] >= 0.1), color='orange', alpha=0.3, label='>10% Return')
    plt.fill_between(operation_df['time'], lower_bound, -0.1, where=(operation_df['rate_of_return'] <= -0.1), color='blue', alpha=0.3, label='<-10% Return')
    plt.title('Rate of Return')
    plt.xlabel('Date')
    plt.ylabel('Rate of Return')
    plt.legend()

    # 设置日期格式为按月显示
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'results/{id}_{name}_DQN.png')
    plt.close()
    print(f'已保存 {id}_{name}_DQN.png')
    # 算出收益率 夏普比率 最大回撤
    # 在best_operations提取出每日的总资产
    total_values = [op['total_value'] for op in best_operations]
    total_values = np.array(total_values)

    def max_drawdown(arr):
        # 计算最大回撤率
        peek = 0
        drawdown = 0
        for i in range(1, len(arr)):
            if arr[i] > arr[peek]:
                peek = i
            else:
                dd = (arr[peek] - arr[i]) / arr[peek]
                drawdown = max(drawdown, dd)
        return drawdown

    def sharpe_ratio(arr):
        # 计算夏普比率
        returns = np.diff(arr) / arr[:-1]
        return (np.mean(returns) * 252 - 0.01) / (np.std(returns) * np.sqrt(252))

    profit_rate = calculate_profit(best_operations)
    max_drawdown_rate = max_drawdown(total_values)
    sharpe = sharpe_ratio(total_values)
    # print(f"收益率: {profit_rate:.2f}%")
    # print(f"最大回撤率: {max_drawdown_rate*100:.2f}%")
    # print(f"夏普比率: {sharpe:.4f}")

    # 添加到结果文件
    with open(f'results/{id}_{name}.csv', 'a') as f:
        # 策略名称, 收益率, 最大回撤率, 夏普比率, 区间涨幅
        f.write(f'DQN, {profit_rate:.2f}, {max_drawdown_rate*100:.2f}, {sharpe:.4f}, {base_return*100}\n')