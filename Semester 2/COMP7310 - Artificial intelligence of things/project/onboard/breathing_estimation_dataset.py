import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt, welch, correlate, butter, filtfilt, find_peaks, welch, sosfiltfilt
from pathlib import Path

# 全局配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
fs = 100


def parse_csi_data(csi_str):
    """解析CSI字符串为幅度值"""
    try:
        # 清洗数据并转换为复数
        csi_str = str(csi_str).strip('[]"')
        elements = [float(x) for x in csi_str.split(',') if x.strip()]

        # 验证数据完整性
        if len(elements) < 64 or len(elements) % 2 != 0:
            raise ValueError(f"CSI数据长度异常: 需要至少64个元素（实虚数对），当前{len(elements)}个")

        # 构造复数对并计算幅度
        magnitudes = [abs(complex(real, imag))
                      for real, imag in zip(elements[1::2], elements[::2])]
        return magnitudes
    except Exception as e:
        print(f"CSI解析错误: {str(e)}")
        return []


def autocorrelation(x):
    x = x - np.nanmean(x)  # 去除直流分量
    x = np.nan_to_num(x)

    cutoff_freq = [0.1, 0.5]
    b, a = butter(5, cutoff_freq, btype='band', fs=fs)
    x_filt = filtfilt(b, a, x)

    # sos = butter(5, cutoff_freq, "band", fs=fs, output="sos")
    # x_filt = sosfiltfilt(sos, x)

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


def process_gt_file(gt_file):
    try:
        gt_df = pd.read_csv(gt_file, sep=',', header=0)

        if len(gt_df.columns) == 1 and ',' in gt_df.columns[0]:
            cols = gt_df.columns[0].split(',')
            gt_df = pd.DataFrame(gt_df.iloc[:, 0].str.split(',').tolist(), columns=cols)


        first_col = gt_df.columns[0]

        bpm_series = pd.to_numeric(gt_df[first_col], errors='coerce').dropna()

        return bpm_series.values  # 去除前14个数据点

    except Exception as e:
        print(f"\n GT文件处理失败: {gt_file.name}")


def visualize_results(pred, gt, filename, save_dir):
    plt.figure(figsize=(18, 12))

    # 时间序列对比
    plt.subplot(2, 2, (1, 3))
    plt.plot(pred, 'b-', label='预测值', alpha=0.7, marker='o', markersize=4)
    plt.plot(gt, 'r--', label='真实值', alpha=0.7)
    plt.title(f'呼吸率对比 [{filename}]')
    plt.xlabel('时间窗序号')
    plt.ylabel('BPM')
    plt.legend()
    plt.grid(True)

    # 误差分布
    plt.subplot(2, 2, 2)
    errors = np.abs(pred - gt)
    plt.hist(errors, bins=20, color='green', alpha=0.7)
    plt.title(f'误差分布 (MAE: {np.nanmean(errors):.2f} BPM)')
    plt.xlabel('绝对误差')
    plt.ylabel('频次')

    # 散点图分析
    plt.subplot(2, 2, 4)
    plt.scatter(gt, pred, c=errors, cmap='viridis', alpha=0.6)
    plt.colorbar(label='绝对误差')
    plt.plot([0, 30], [0, 30], 'r--')
    plt.title('预测值 vs 真实值')
    plt.xlabel('真实BPM')
    plt.ylabel('预测BPM')

    # 保存图表
    plot_path = save_dir / f"{filename.replace('.csv', '')}_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存至: {plot_path}")


def process_dataset(base_path):
    data_dir = Path(base_path) / "breathing_rate" / "evaluation"
    results = []

    plot_dir = data_dir / "visualization"
    plot_dir.mkdir(exist_ok=True, parents=True)

    for csi_file in data_dir.glob('CSI*.csv'):
        try:
            print("\n" + "=" * 60)
            print(f"正在处理文件: {csi_file.name}")

            csi_df = pd.read_csv(csi_file)

            # 解析CSI幅度数据
            csi_amplitudes = csi_df['data'].apply(
                lambda x: parse_csi_data(x) or [np.nan]
            ).apply(np.nanmean)

            # 填充CSI前14s
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

            # =================================================================
            # GT数据处理
            # =================================================================
            csi_prefix = csi_file.stem[3:]
            gt_file = data_dir / f"gt_{csi_prefix}.csv"
            if not gt_file.exists():
                gt_file = data_dir / f"gt_CSI{csi_prefix}.csv"

            if not gt_file.exists():
                raise FileNotFoundError(f"未找到对应的GT文件: {gt_file}")

            gt_bpm = process_gt_file(gt_file)

            min_length = min(len(pred_bpm), len(gt_bpm))
            aligned_pred = np.array(pred_bpm[:min_length], dtype=float)
            aligned_gt = np.array(gt_bpm[:min_length], dtype=float)

            result_df = pd.DataFrame({
                '时间窗序号': np.arange(min_length) + 1,
                '预测BPM': aligned_pred,
                '真实BPM': aligned_gt,
                '绝对误差': np.abs(aligned_pred - aligned_gt)
            })

            # 保存CSV
            csv_path = plot_dir / f"{csi_file.stem}_results.csv"
            result_df.to_csv(csv_path, index=False)
            print(f"结果文件已保存至: {csv_path}")

            print("\n完整数据对比:")
            print(result_df.to_string(
                index=False,
                formatters={
                    '预测BPM': lambda x: f"{x:.2f}",
                    '真实BPM': lambda x: f"{x:.2f}",
                    '绝对误差': lambda x: f"{x:.2f}"
                }
            ))

            # 可视化
            visualize_results(aligned_pred, aligned_gt, csi_file.name, plot_dir)
            # 计算评估指标
            valid_mask = ~np.isnan(aligned_pred) & ~np.isnan(aligned_gt)
            abs_errors = np.abs(aligned_pred[valid_mask] - aligned_gt[valid_mask])

            results.append({
                '文件': csi_file.name,
                'mMAE': np.median(abs_errors),
                'MAE': np.nanmean(abs_errors),
                '数据点数': valid_mask.sum(),
                '最大误差': np.nanmax(abs_errors)
            })

        except Exception as e:
            print(f"\n⚠️ 处理失败: {str(e)}")
            continue

        if results:
            summary_df = pd.DataFrame(results)
            print("\n\n" + "=" * 60)
            print("汇总评估结果:")
            print(summary_df.to_string(index=False))
            print("\n总体统计:")
            print(f"平均mMAE: {summary_df['mMAE'].mean():.2f} ± {summary_df['mMAE'].std():.2f} BPM")
            print(f"平均MAE: {summary_df['MAE'].mean():.2f} ± {summary_df['MAE'].std():.2f} BPM")
            print(f"总数据点数: {summary_df['数据点数'].sum()}")
            print(f"最大误差: {summary_df['最大误差'].max():.2f} BPM")

if __name__ == '__main__':
    dataset_path = 'benchmark'

    process_dataset(dataset_path)