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

    env_1 = StockTradingEnv(df.to_dict(orient='records'),buy_ratio=0.5,sell_ratio=0.5)
    env_2 = StockTradingEnv(df.to_dict(orient='records'),buy_ratio=0.1,sell_ratio=0.9)

    state_size = env_1.observation_space.shape[0]
    action_size = env_1.action_space.n
    agent_1 = DQNAgent(state_size, action_size)
    agent_2 = DQNAgent(state_size, action_size)

    # 如果加载训练次数，则加载模型
    if start_episode > 0:
        agent_1.load_model(start_episode)

    # 训练智能体 并且返回每日操作日志
    episodes = epoch
    print(f"开始训练 {id} {name} {episodes} 次")


    best_operations_1 = plot_profit(id, name, df, base_return, start_episode, env_1, agent_1, episodes,'Aggressive')
    best_operations_2 = plot_profit(id, name, df, base_return, start_episode, env_2, agent_2, episodes,'Conservative')

    # 算出收益率 夏普比率 最大回撤
    # 在best_operations提取出每日的总资产
    total_values_1 = [op['total_value'] for op in best_operations_1]
    total_values_1 = np.array(total_values_1)

    total_values_2 = [op['total_value'] for op in best_operations_2]
    total_values_2 = np.array(total_values_2)

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

    profit_rate_1 = calculate_profit(best_operations_1)
    max_drawdown_rate_1 = max_drawdown(total_values_1)
    sharpe_1 = sharpe_ratio(total_values_1)

    profit_rate_2 = calculate_profit(best_operations_2)
    max_drawdown_rate_2 = max_drawdown(total_values_2)
    sharpe_2 = sharpe_ratio(total_values_2)
    # print(f"收益率: {profit_rate:.2f}%")
    # print(f"最大回撤率: {max_drawdown_rate*100:.2f}%")
    # print(f"夏普比率: {sharpe:.4f}")

    # 添加到结果文件
    with open(f'results/{id}_{name}.csv', 'a') as f:
        # 策略名称, 收益率, 最大回撤率, 夏普比率, 区间涨幅
        f.write(f'DQN_Aggressive, {profit_rate_1:.2f}, {max_drawdown_rate_1*100:.2f}, {sharpe_1:.4f}, {base_return*100}\n')
        f.write(f'DQN_Conservative, {profit_rate_2:.2f}, {max_drawdown_rate_2*100:.2f}, {sharpe_2:.4f}, {base_return*100}\n')

def plot_profit(id, name, df, base_return, start_episode, env_1, agent_1, episodes,postfix):
    all_daily_operations_1 = train_agent(env_1, agent_1, episodes=episodes, batch_size=64, start_episode=start_episode, base_return=base_return)
    best_operations_1 = max(all_daily_operations_1, key=lambda x: calculate_profit(x))
    import matplotlib.dates as mdates

    # 输出最佳利率
    operation_df_1 = pd.DataFrame(best_operations_1)
    operation_df_1.to_csv(f'{id}_{name}_operation_{postfix}.csv')
    df.to_csv(f'{id}_{name}_price.csv')
    operation_df_1['time'] = pd.to_datetime(operation_df_1['time'])
    profit_rate_1 = calculate_profit(best_operations_1)


    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.figure(figsize=(14, 7))
    # 计算数据划分的索引
    split_index = int(len(df) * 0.75)

    # 绘制前四分之三的数据（蓝色线条）
    plt.plot(df['date'][:split_index], df['close'][:split_index], label='Closing Price (Train)', color='blue')

    # 绘制后四分之一的数据（橙色线条）
    plt.plot(df['date'][split_index:], df['close'][split_index:], label='Closing Price (Test)', color='red')

    plt.title(f'DQN - {id} {name} Profit Rate: {profit_rate_1:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()

    # 设置日期格式为按月显示
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # 自动选择合适的日期间隔
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 按 "年-月" 格式显示日期
    plt.xticks(rotation=45)  # 旋转日期标签，避免重叠

    plt.tight_layout()
    plt.savefig(f'results/{id}_{name}_DQN_Profit.png')
    plt.close()
    print(f'已保存 {id}_{name}_DQN_Profit.png')

    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.figure(figsize=(14,7))

    # 子图1: 股票收盘价和买卖信号
    plt.subplot(3, 1, 1)
    plt.plot(df['date'], df['close'], label='Closing Price')  
    plt.scatter(operation_df_1[operation_df_1['action'] == 'buy']['time'], operation_df_1[operation_df_1['action'] == 'buy']['price'], marker='^', c='g', label='Buy')
    plt.scatter(operation_df_1[operation_df_1['action'] == 'sell']['time'], operation_df_1[operation_df_1['action'] == 'sell']['price'], marker='v', c='r', label='Sale')
    plt.title(f'DQN - {id} {name} Profit Rate: {profit_rate_1:.2f}%')
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
    operation_df_1['total_value'] = operation_df_1['cash'] + operation_df_1['stock_value']
    plt.plot(operation_df_1['time'], operation_df_1['total_value'], label='Total Amount')
    plt.plot(operation_df_1['time'], operation_df_1['stock_value'], label='Position')
    plt.title('Daily Holdings and Total Value (Best Operations)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()

    # 收益率曲线 低于0的部分标记为绿色 高于0的部分标记为红色 低于10%的部分标记为蓝色 高于10%的部分标记为橙色
    plt.subplot(3, 1, 3)
    operation_df_1['rate_of_return'] = (operation_df_1['total_value'] - initial_cash) / initial_cash
    plt.plot(operation_df_1['time'], operation_df_1['rate_of_return'], label='Rate of Return')
    # 首先找到上界和下界 作为图标的纵轴
    upper_bound = operation_df_1['rate_of_return'].max()
    lower_bound = operation_df_1['rate_of_return'].min()
    plt.fill_between(operation_df_1['time'], lower_bound, 0, where=(operation_df_1['rate_of_return'] <= 0), color='green', alpha=0.3, label='Negative Return')
    plt.fill_between(operation_df_1['time'], 0, upper_bound, where=(operation_df_1['rate_of_return'] > 0) & (operation_df_1['rate_of_return'] < 0.1), color='red', alpha=0.3, label='Positive Return')
    plt.fill_between(operation_df_1['time'], 0.1, upper_bound, where=(operation_df_1['rate_of_return'] >= 0.1), color='orange', alpha=0.3, label='>10% Return')
    plt.fill_between(operation_df_1['time'], lower_bound, -0.1, where=(operation_df_1['rate_of_return'] <= -0.1), color='blue', alpha=0.3, label='<-10% Return')
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
    plt.savefig(f'results/{id}_{name}_DQN_{postfix}.png')
    plt.close()
    print(f'已保存 {id}_{name}_DQN_{postfix}.png')
    return best_operations_1