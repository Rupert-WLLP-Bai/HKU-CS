import akshare as ak
import pandas as pd

def get_hs300_components() -> pd.DataFrame:
    """
    获取沪深300成分股列表。

    返回:
        pd.DataFrame: 包含沪深300成分股代码和名称的 DataFrame。
    """
    return ak.index_stock_cons(symbol="000300")


def get_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定股票的历史数据。

    参数:
        stock_code (str): 股票代码。
        start_date (str): 数据开始日期，格式为"YYYY-MM-DD"。
        end_date (str): 数据结束日期，格式为"YYYY-MM-DD"。

    返回:
        pd.DataFrame: 指定股票的历史数据，包含日期、开盘、收盘、最高、最低、成交量、换手率等字段。
    """
    df = ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="")
    # 保留需要的字段
    df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量', '换手率']]
    
    # 重命名列名为 Backtrader 标准字段
    df.rename(columns={
        '日期': 'datetime',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '换手率': 'turnover'
    }, inplace=True)

    # 将 datetime 转换为时间类型并设为索引
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    return df


def get_hs300_sample(stock_count: int, start_date: str, end_date: str) -> dict:
    """
    随机抽取沪深300成分股并获取其历史数据。

    参数:
        stock_count (int): 随机抽取的股票数量。
        start_date (str): 数据开始日期，格式为"YYYY-MM-DD"。
        end_date (str): 数据结束日期，格式为"YYYY-MM-DD"。

    返回:
        dict: 包含随机抽取的股票代码和其历史数据的字典，键为股票代码，值为对应的 DataFrame。
    """
    # 获取沪深300指数成分股
    hs300_origin = get_hs300_components()
    
    # 检查是否超过成分股总数
    if stock_count > len(hs300_origin):
        raise ValueError("指定的股票数量超过沪深300成分股总数！")
    
    # 随机抽取指定数量的股票
    hs300_sample = hs300_origin.sample(stock_count)
    
    # 查询每只股票的历史数据
    data = {}
    for row in hs300_sample.iterrows():
        code = row[1]['品种代码']
        name = row[1]['品种名称']
        print(f"Fetching data for: {name} ({code})")
        data[code] = get_stock_data(stock_code=code, start_date=start_date, end_date=end_date)
    
    return data

def get_stock_name(id: str) -> str:
    """
    获取指定股票代码的股票名称。

    参数:
        symbol (str): 股票代码。

    返回:
        str: 股票名称。
    """
    return ak.stock_individual_info_em(symbol=id)['value'][1]