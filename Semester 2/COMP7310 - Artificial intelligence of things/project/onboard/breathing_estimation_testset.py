import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
fs = 100

def parse_csi_data(csi_str):
    """解析CSI字符串为幅度值"""
    try:

        csi_str = str(csi_str).strip('[]"')
        elements = [float(x) for x in csi_str.split(',') if x.strip()]

        magnitudes = [abs(complex(real, imag))
                      for real, imag in zip(elements[::2], elements[1::2])]
        return magnitudes
    except Exception as e:
        print(f"CSI解析错误: {str(e)}")
        return []


def autocorrelation(x):
    x = x - np.nanmean(x)  # 去除直流分量
    x = np.nan_to_num(x)

    b, a = butter(5, [0.1, 0.5], btype='band', fs=fs)
    x_filt = filtfilt(b, a, x)

    # 标准化
    x_norm = x_filt / (np.std(x_filt) + 1e-9)

    # 自相关计算
    corr = np.correlate(x_norm, x_norm, mode='full')
    corr = corr[len(corr) // 2:]

    return corr

def estimate_breathing_rate(signal, fs=100):

    if len(signal) < 15*fs:
        return np.nan

    try:
        acf = autocorrelation(signal)
        peaks, props = find_peaks(acf)

        period = peaks[0] / fs
        br = 60 / period

        return br

    except Exception as e:
        print(f"处理异常: {str(e)}")
        return np.nan


def process_testset(base_path):
    data_dir = Path(base_path) / "breathing_rate" / "test"
    results = []

    plot_dir = data_dir / "visualization"
    plot_dir.mkdir(exist_ok=True, parents=True)

    for csi_file in data_dir.glob('CSI*.csv'):
        print("\n" + "=" * 60)
        print(f"正在处理文件: {csi_file.name}")
        csi_df = pd.read_csv(csi_file)

        # 解析CSI数据为幅值序列
        csi_amplitudes = csi_df['data'].apply(
            lambda x: parse_csi_data(x) or [np.nan]
        ).apply(np.nanmean)

        # CSI数据向前填充前14s
        def padding(data, pad_samples):
            front_pad = np.flip(data[:pad_samples])
            end_pad = np.flip(data[-pad_samples:])
            return np.concatenate([front_pad, data, end_pad])

        pad_samples = 14 * fs
        padded_data = padding(csi_amplitudes.interpolate().values, pad_samples)

        window_size = 15 * fs
        pred_bpm = []

        for start in range(0, len(padded_data) - window_size + 1, fs):
            window = padded_data[start: start + window_size]
            bpm = estimate_breathing_rate(window, fs)
            pred_bpm.append(bpm)
        print(f"生成预测值: {len(pred_bpm)}个")

        # 保存结果为CSV
        time_windows = list(range(1, 1 + len(pred_bpm)))
        result_df = pd.DataFrame({
            '时间': time_windows,
            'BPM': np.round(pred_bpm, 2)
        })
        csv_path = plot_dir / f"{csi_file.stem}_results.csv"
        result_df.to_csv(csv_path, index=False)
        print(f"结果CSV已保存至: {csv_path}")

        # 输出图像
        plt.figure(figsize=(12, 6))
        plt.plot(time_windows, pred_bpm, marker='o', linestyle='-', color='blue', label='预测BPM')
        plt.title(f'呼吸率预测结果 - {csi_file.stem}')
        plt.xlabel('时间')
        plt.ylabel('BPM')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        img_path = plot_dir / f"{csi_file.stem}_results.png"
        plt.savefig(img_path, dpi=150)
        plt.close()
        print(f"可视化图像已保存至: {img_path}")

    return results

if __name__ == '__main__':
    dataset_path = 'benchmark'

    process_testset(dataset_path)