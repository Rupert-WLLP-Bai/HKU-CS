# Author: Bai Junhao
# Date: 2024-11-17 21:37:39

from data.stock_data import get_stock_data, get_stock_name
from strategies.strategy import DualMovingAverageConservative, DualMovingAverageAggressive, RSIConservative, RSIAggressive, VolatilityConservative, VolatilityAggressive
import backtrader as bt
import backtrader.feeds as btfeeds
import yaml
import pandas as pd
import matplotlib.pyplot as plt

# 读取配置文件
user_config = yaml.safe_load(open('config/user.yaml', 'r', encoding='utf-8'))
initial_cash = user_config['user']['initial_cash']
stock_config = yaml.safe_load(open('config/stock.yaml', 'r', encoding='utf-8'))
start_date = stock_config['start_date']
end_date = stock_config['end_date']
stock_list = stock_config['stock_list']

# 策略列表
strategies = [
    ("DualMovingAverageConserative", DualMovingAverageConservative),
    ("DualMovingAverageAggressive", DualMovingAverageAggressive),
    ("RSIConserative", RSIConservative),
    ("RSIAggressive", RSIAggressive),
    ("VolatilityConserative", VolatilityConservative),
    ("VolatilityAggressive", VolatilityAggressive),
]

def backtest(stock_code):
    name = get_stock_name(stock_code)
    test_data = get_stock_data(stock_code, start_date, end_date)
    # 存储结果
    results = []
    # 单独运行每个策略
    for strategy_name, strategy_class in strategies:
        # 创建 Cerebro 实例
        cerebro = bt.Cerebro()
        
        # 加载数据
        data = btfeeds.PandasData(dataname=test_data)
        cerebro.adddata(data)
        
        # 添加策略
        cerebro.addstrategy(strategy_class)
        
        # 设置初始资金
        cerebro.broker.set_cash(initial_cash)
        
        # 设置手续费
        cerebro.broker.setcommission(commission=0.0005)
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # 运行回测
        result = cerebro.run()
        strat = result[0]
        
        # 获取结果
        final_value = cerebro.broker.getvalue()
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        
        # 保存结果
        results.append({
            'strategy': strategy_name,
            'final_value': final_value,
            'sharpe': sharpe,
            'drawdown': drawdown,
            'returns': returns,
            'cerero_instance': cerebro,
        })

    # 输出结果
    for res in results:
        # print(f"策略: {res['strategy']}")
        # print(f"初始资金: {initial_cash:.2f}")
        # print(f"结束资金: {res['final_value']:.2f}")
        # print(f"夏普比率: {res['sharpe']}")
        # print(f"最大回撤: {res['drawdown']}")
        # print(f"收益率: {res['returns']}")
        # print("-" * 50)
        
        plt.switch_backend('Agg') # 设置不显示图表
        cerebro_res = res['cerero_instance']
        # fig = cerebro_res.plot(style='candle')
        # show = fig[0][0]
        # show.set_size_inches(30,15)
        # show.savefig(f'{id}_{res["strategy"]}.png')
        # fig.close()

    # 策略数据
    strategy_names = [res['strategy'] for res in results]
    returns = [res['returns']['rtot'] * 100 for res in results]  # 转换为百分比
    max_drawdowns = [res['drawdown']['max']['drawdown'] for res in results]
    sharpe_ratios = [res['sharpe']['sharperatio'] for res in results]

    # 保存数值到csv
    df = pd.DataFrame({
        '策略': strategy_names,
        '收益率 (%)': returns,
        '最大回撤 (%)': max_drawdowns,
        '夏普比率': sharpe_ratios,
        '区间涨幅': [test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1] * len(strategies)
    })
    df.to_csv(f'results/{stock_code}_{name}.csv', index=False)
    print(f"已保存backtest结果到results/{stock_code}_{name}.csv")

import threading
def backtest_all():
    threads = []
    for stock in stock_list:
        code = stock['id']
        name = stock['name']
        # 创建线程
        t = threading.Thread(target=backtest, args=(code,))
        threads.append(t)
        t.start()
        print(f"开始回测 {code} {name}")
    for t in threads:
        t.join()
    print("所有回测完成")
    
        
        