import backtrader as bt
import pandas as pd
from strategies.base import BaseConservativeStrategy, BaseAggressiveStrategy
import backtrader.indicators as btind
import yaml

# 读取配置文件
with open('config/user.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    position = config['position_ratio']
    max_position_conservative = position['max_conservative']
    min_position_conservative = position['min_conservative']
    max_position_aggressive = position['max_aggressive']
    min_position_aggressive = position['min_aggressive']
class DualMovingAverageConservative(BaseConservativeStrategy):
    params = (
        ('sma1', 20),
        ('sma2', 50),
        ('max_position_ratio', max_position_conservative),
        ('min_position_ratio', min_position_conservative),
    )

    def __init__(self):
        super().__init__()
        self.sma1 = btind.SimpleMovingAverage(self.data, period=self.params.sma1)
        self.sma2 = btind.SimpleMovingAverage(self.data, period=self.params.sma2)
        self.crossover = btind.CrossOver(self.sma1, self.sma2)
        
        self.asset_log = [] # 用于记录每天的总资产

    def next(self):
        # 记录当前的总资产
        current_value = self.broker.getvalue()
        self.asset_log.append({
            'date': self.data.datetime.date(0),  # 获取当前时间
            'total_value': current_value
        })
        if self.crossover > 0:
            self.dynamic_buy()
        elif self.crossover < 0:
            self.dynamic_sell()
            
    def stop(self):
        # 回测结束后，将资产记录保存为 CSV
        df = pd.DataFrame(self.asset_log)
        df.to_csv(f'results/{self.__class__.__name__}_assets.csv', index=False)
        print(f"资产变化记录已保存: results/{self.__class__.__name__}_assets.csv")
            
class DualMovingAverageAggressive(BaseAggressiveStrategy):
    params = (
        ('sma1', 5),
        ('sma2', 10),
        ('max_position_ratio', max_position_aggressive),
        ('min_position_ratio', min_position_aggressive),
    )

    def __init__(self):
        super().__init__()
        self.sma1 = btind.SimpleMovingAverage(self.data, period=self.params.sma1)
        self.sma2 = btind.SimpleMovingAverage(self.data, period=self.params.sma2)
        self.crossover = btind.CrossOver(self.sma1, self.sma2)
        self.asset_log = [] # 用于记录每天的总资产
        
    def next(self):
        # 记录当前的总资产
        current_value = self.broker.getvalue()
        self.asset_log.append({
            'date': self.data.datetime.date(0),  # 获取当前时间
            'total_value': current_value
        })
        if self.crossover > 0:
            self.dynamic_buy()
        elif self.crossover < 0:
            self.dynamic_sell()
            
    def stop(self):
        # 回测结束后，将资产记录保存为 CSV
        df = pd.DataFrame(self.asset_log)
        df.to_csv(f'results/{self.__class__.__name__}_assets.csv', index=False)
        print(f"资产变化记录已保存: results/{self.__class__.__name__}_assets.csv")
            
class RSIConservative(BaseConservativeStrategy):
    params = (
        ('rsi_low', 30),
        ('rsi_high', 70),
        ('max_position_ratio', max_position_conservative),
        ('min_position_ratio', min_position_conservative),
    )

    def __init__(self):
        super().__init__()
        self.rsi = btind.RSI(self.data)
        self.asset_log = [] # 用于记录每天的总资产
        
    def next(self):
        # 记录当前的总资产
        current_value = self.broker.getvalue()
        self.asset_log.append({
            'date': self.data.datetime.date(0),  # 获取当前时间
            'total_value': current_value
        })
        if self.rsi < self.params.rsi_low:
            self.dynamic_buy()
        elif self.rsi > self.params.rsi_high:
            self.dynamic_sell()
            
    def stop(self):
        # 回测结束后，将资产记录保存为 CSV
        df = pd.DataFrame(self.asset_log)
        df.to_csv(f'results/{self.__class__.__name__}_assets.csv', index=False)
        print(f"资产变化记录已保存: results/{self.__class__.__name__}_assets.csv")
            
class RSIAggressive(BaseAggressiveStrategy):
    params = (
        ('rsi_low', 40),
        ('rsi_high', 60),
        ('max_position_ratio', max_position_aggressive),
        ('min_position_ratio', min_position_aggressive),
    )

    def __init__(self):
        super().__init__()
        self.rsi = btind.RSI(self.data)
        self.asset_log = [] # 用于记录每天的总资产
        
    def next(self):
        # 记录当前的总资产
        current_value = self.broker.getvalue()
        self.asset_log.append({
            'date': self.data.datetime.date(0),  # 获取当前时间
            'total_value': current_value
        })
        if self.rsi < self.params.rsi_low:
            self.dynamic_buy()
        elif self.rsi > self.params.rsi_high:
            self.dynamic_sell()
            
    def stop(self):
        # 回测结束后，将资产记录保存为 CSV
        df = pd.DataFrame(self.asset_log)
        df.to_csv(f'results/{self.__class__.__name__}_assets.csv', index=False)
        print(f"资产变化记录已保存: results/{self.__class__.__name__}_assets.csv")            

            
class VolatilityConservative(BaseConservativeStrategy):
    params = (
        ('atr_period', 10),
        ('atr_multiplier', 1.0),
        ('max_position_ratio', max_position_conservative),
        ('min_position_ratio', min_position_conservative),
    )

    def __init__(self):
        super().__init__()
        self.atr = btind.ATR(self.data, period=self.params.atr_period)
        self.asset_log = [] # 用于记录每天的总资产
        
    def next(self):
        # 记录当前的总资产
        current_value = self.broker.getvalue()
        self.asset_log.append({
            'date': self.data.datetime.date(0),  # 获取当前时间
            'total_value': current_value
        })
        if self.data.close > self.data.high[-1] + self.atr * self.params.atr_multiplier:
            self.dynamic_buy()
        elif self.data.close < self.data.low[-1] - self.atr * self.params.atr_multiplier:
            self.dynamic_sell()
            
    def stop(self):
        # 回测结束后，将资产记录保存为 CSV
        df = pd.DataFrame(self.asset_log)
        df.to_csv(f'results/{self.__class__.__name__}_assets.csv', index=False)
        print(f"资产变化记录已保存: results/{self.__class__.__name__}_assets.csv")
            
class VolatilityAggressive(BaseAggressiveStrategy):
    params = (
        ('atr_period', 10),
        ('atr_multiplier', 0.5),
        ('max_position_ratio', max_position_aggressive),
        ('min_position_ratio', min_position_aggressive),
    )

    def __init__(self):
        super().__init__()
        self.atr = btind.ATR(self.data, period=self.params.atr_period)
        self.asset_log = []
        
    def next(self):
        # 记录当前的总资产
        current_value = self.broker.getvalue()
        self.asset_log.append({
            'date': self.data.datetime.date(0),  # 获取当前时间
            'total_value': current_value
        })
        if self.data.close > self.data.high[-1] + self.atr * self.params.atr_multiplier:
            self.dynamic_buy()
        elif self.data.close < self.data.low[-1] - self.atr * self.params.atr_multiplier:
            self.dynamic_sell()
            
    def stop(self):
        # 回测结束后，将资产记录保存为 CSV
        df = pd.DataFrame(self.asset_log)
        df.to_csv(f'results/{self.__class__.__name__}_assets.csv', index=False)
        print(f"资产变化记录已保存: results/{self.__class__.__name__}_assets.csv")