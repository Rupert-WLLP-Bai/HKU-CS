import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
stock_config = yaml.safe_load(open('config/stock.yaml', 'r', encoding='utf-8'))
start_date = stock_config['start_date']
end_date = stock_config['end_date']
stock_list = stock_config['stock_list']

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_results(data, id, name, base_return):
    # 提取数据
    strategy_names = data['策略']
    returns = data['收益率 (%)']
    max_drawdowns = data['最大回撤 (%)']
    sharpe_ratios = data['夏普比率']

    # 设置图表布局
    x = np.arange(len(strategy_names))  # 策略数量
    width = 0.25  # 柱状图宽度

    fig, ax = plt.subplots(figsize=(12,8))

    # 绘制柱状图
    bar1 = ax.bar(x - width, returns, width, label='收益率 (%)', color='tab:blue')
    bar2 = ax.bar(x, max_drawdowns, width, label='最大回撤 (%)', color='tab:orange')
    bar3 = ax.bar(x + width, sharpe_ratios, width, label='夏普比率', color='tab:green')

    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.2f}', ha='center', va='bottom')

    add_labels(bar1)
    add_labels(bar2)
    add_labels(bar3)
    
    # 添加一条基准线 表示区间涨幅
    # 再添加具体的数值标签
    if base_return > 0:
        ax.axhline(y=base_return * 100, color='r', linestyle='--', label='区间涨幅')
        ax.text(x[-1] + 0.5, base_return * 100 + 1, f'{base_return * 100:.2f}%', ha='left', va='center', color='r')
    else:
        ax.axhline(y=base_return * 100, color='g', linestyle='--', label='区间涨幅')
        ax.text(x[-1] + 0.5, base_return * 100 - 1, f'{base_return * 100:.2f}%', ha='left', va='center', color='g')
    # 添加标题和标签
    ax.set_title(f'{id} {name}', fontsize=16)
    ax.set_xlabel('策略', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax.legend()

    # 调整布局并显示图表
    plt.tight_layout()
    plt.savefig(f'results/{id}_{name}.png')
    # plt.show()
    
def read_data(id, name):
    # 读取数据
    data = pd.read_csv(f'./results/{id}_{name}.csv')
    return data

def plot_all():
    for stock in stock_list:
        id = stock['id']
        name = stock['name']
        data = read_data(id, name)
        plot_results(data, id, name, data['区间涨幅'][0])
