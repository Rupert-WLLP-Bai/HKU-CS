from backtest import backtest_all
from train import train_all
from utils.plot import plot_all

if __name__ == '__main__':
    backtest_all()
    train_all(epoch=100)
    plot_all()
    
    