# Author: Bai Junhao
# Date: 2024-11-17 21:37:39

import backtrader as bt

class BaseConservativeStrategy(bt.Strategy):
    params = (
        ('max_position_ratio', 0.7),  # 默认值，可在子类中覆盖
        ('min_position_ratio', 0.1),  # 默认值，可在子类中覆盖
        ('adjust_ratio', 0.1),        # 每次调整仓位的比例
        ('min_trade_size', 1),        # 最小交易单位
    )

    def __init__(self):
        self.max_position_ratio = self.params.max_position_ratio
        self.min_position_ratio = self.params.min_position_ratio
        self.adjust_ratio = self.params.adjust_ratio
        self.min_trade_size = self.params.min_trade_size

    def get_position_ratio(self):
        """计算当前仓位比例"""
        total_value = self.broker.getvalue()  # 当前账户总资产
        position_value = self.broker.getposition(self.data).size * self.data.close[0]
        return position_value / total_value

    def dynamic_buy(self):
        """动态加仓逻辑"""
        position_ratio = self.get_position_ratio()
        if position_ratio < self.max_position_ratio:
            available_cash = self.broker.getcash()
            max_investment = self.broker.getvalue() * self.max_position_ratio
            target_investment = position_ratio * self.broker.getvalue() + (max_investment - position_ratio * self.broker.getvalue()) * self.adjust_ratio
            investment = min(target_investment, available_cash)
            size = int(investment // self.data.close[0])
            if size >= self.min_trade_size:
                self.buy(size=size)

    def dynamic_sell(self):
        """动态减仓逻辑"""
        position_ratio = self.get_position_ratio()
        if position_ratio > self.min_position_ratio:
            position_value = self.broker.getposition(self.data).size * self.data.close[0]
            min_investment = self.broker.getvalue() * self.min_position_ratio
            target_investment = position_value - (position_value - min_investment) * self.adjust_ratio
            size = int((position_value - target_investment) // self.data.close[0])
            if size >= self.min_trade_size:
                self.sell(size=size)


class BaseAggressiveStrategy(bt.Strategy):
    params = (
        ('max_position_ratio', 1.0),  # 默认值，可在子类中覆盖
        ('min_position_ratio', 0.3),  # 默认值，可在子类中覆盖
        ('adjust_ratio', 0.2),        # 每次调整仓位的比例
        ('min_trade_size', 1),        # 最小交易单位
    )

    def __init__(self):
        self.max_position_ratio = self.params.max_position_ratio
        self.min_position_ratio = self.params.min_position_ratio
        self.adjust_ratio = self.params.adjust_ratio
        self.min_trade_size = self.params.min_trade_size

    def get_position_ratio(self):
        """计算当前仓位比例"""
        total_value = self.broker.getvalue()  # 当前账户总资产
        position_value = self.broker.getposition(self.data).size * self.data.close[0]
        return position_value / total_value

    def dynamic_buy(self):
        """动态加仓逻辑"""
        position_ratio = self.get_position_ratio()
        if position_ratio < self.max_position_ratio:
            available_cash = self.broker.getcash()
            max_investment = self.broker.getvalue() * self.max_position_ratio
            target_investment = position_ratio * self.broker.getvalue() + (max_investment - position_ratio * self.broker.getvalue()) * self.adjust_ratio
            investment = min(target_investment, available_cash)
            size = int(investment // self.data.close[0])
            if size >= self.min_trade_size:
                self.buy(size=size)

    def dynamic_sell(self):
        """动态减仓逻辑"""
        position_ratio = self.get_position_ratio()
        if position_ratio > self.min_position_ratio:
            position_value = self.broker.getposition(self.data).size * self.data.close[0]
            min_investment = self.broker.getvalue() * self.min_position_ratio
            target_investment = position_value - (position_value - min_investment) * self.adjust_ratio
            size = int((position_value - target_investment) // self.data.close[0])
            if size >= self.min_trade_size:
                self.sell(size=size)