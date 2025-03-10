import cupy as cp
import cupyx.scipy.signal as csignal
from cupyx.scipy.signal import hilbert
import arlpy.uwapm as pm
import numpy as np
from scipy.signal import butter
import argparse
from tqdm import tqdm
import os

# 配置声速剖面和环境的代码与原CPU版本相同
ssp = [                                 
  [  0.0 , 1548.52],[200.0 , 1530.29],[400.0 , 1517.78],[600.0 , 1509.49 ],  
  [800.0 , 1504.30],[1000.0 , 1501.38],[1200.0 ,1500.14],[1400.0 ,  1500.12],  
  [1600.0 ,1501.02],[1800.0 , 1502.57] , [2000.0 , 1504.62],[2200.0 , 1507.02 ],  
  [2400.0 , 1509.69 ],[2600.0 , 1512.55],[2800.0 , 1515.56],[3000.0 , 1518.67 ],
  [ 3200.0 , 1521.85 ],[3400.0 , 1525.10 ],[3600.0 , 1528.38],[3800.0 , 1531.70],
  [4000.0 , 1535.04],[4200.0 , 1538.39],[4400.0 , 1541.76 ],[4600.0 , 1545.14],
  [4800.0 , 1548.52 ],[5000.0 , 1551.91]
]

env = pm.create_env2d(
    depth=5000,
    soundspeed=ssp,
    bottom_soundspeed=1551.91,
    bottom_density=1200,
    bottom_absorption=1.0,
    nbeams=70,
    tx_depth=200,
    rx_depth=400,
    frequency=1500,
    rx_range=60000
)

# 生成Chirp信号（GPU版本）
def generate_chirp(duration, f0, f1, fs):
    t = cp.linspace(0, duration, int(duration * fs), endpoint=False)
    chirp_signal = cp.sin(2 * cp.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    return chirp_signal

# 带通滤波器系数在CPU计算
fs = 4000
fmax = fs / 2
f_start, f_end = 1350 / fmax, 1650 / fmax
b, a = butter(8, [f_start, f_end], 'bandpass')
b_gpu, a_gpu = cp.asarray(b), cp.asarray(a)

# 计算脉冲响应（CPU）
arrivals = pm.compute_arrivals(env)
ir = pm.arrivals_to_impulse_response(arrivals, fs=fs, abs_time=True)
ir_gpu = cp.asarray(ir)  # 转换为CuPy数组

# 生成参考信号（GPU）import cupy as cp
import cupyx.scipy.signal as csignal
from cupyx.scipy.signal import hilbert
import arlpy.uwapm as pm
import numpy as np
from scipy.signal import butter
import argparse
from tqdm import tqdm
import os

# 配置声速剖面和环境的代码与原CPU版本相同
ssp = [
  [  0.0 , 1548.52],[200.0 , 1530.29],[400.0 , 1517.78],[600.0 , 1509.49 ],
  [800.0 , 1504.30],[1000.0 , 1501.38],[1200.0 ,1500.14],[1400.0 ,  1500.12],
  [1600.0 ,1501.02],[1800.0 , 1502.57] , [2000.0 , 1504.62],[2200.0 , 1507.02 ],
  [2400.0 , 1509.69 ],[2600.0 , 1512.55],[2800.0 , 1515.56],[3000.0 , 1518.67 ],
  [ 3200.0 , 1521.85 ],[3400.0 , 1525.10 ],[3600.0 , 1528.38],[3800.0 , 1531.70],
  [4000.0 , 1535.04],[4200.0 , 1538.39],[4400.0 , 1541.76 ],[4600.0 , 1545.14],
  [4800.0 , 1548.52 ],[5000.0 , 1551.91]
]

env = pm.create_env2d(
    depth=5000,
    soundspeed=ssp,
    bottom_soundspeed=1551.91,
    bottom_density=1200,
    bottom_absorption=1.0,
    nbeams=70,
    tx_depth=200,
    rx_depth=400,
    frequency=1500,
    rx_range=60000
)

# 生成Chirp信号（GPU版本）
def generate_chirp(duration, f0, f1, fs):
    t = cp.linspace(0, duration, int(duration * fs), endpoint=False)
    chirp_signal = cp.sin(2 * cp.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    return chirp_signal

# 带通滤波器系数在CPU计算
fs = 4000
fmax = fs / 2
f_start, f_end = 1350 / fmax, 1650 / fmax
b, a = butter(8, [f_start, f_end], 'bandpass')
b_gpu, a_gpu = cp.asarray(b), cp.asarray(a)

# 计算脉冲响应（CPU）
arrivals = pm.compute_arrivals(env)
ir = pm.arrivals_to_impulse_response(arrivals, fs=fs, abs_time=True)
ir_gpu = cp.asarray(ir)  # 转换为CuPy数组

# 生成参考信号（GPU）
duration = 0.5
f0, f1 = 1400, 1600
base_chirp = generate_chirp(duration, f0, f1, fs)
template_gpu = base_chirp[::-1].copy()  # 匹配滤波模板

# 噪声生成函数（GPU）
def awgn_gpu(signal, snr):
    signal_power = cp.sum(cp.abs(signal)**2) / len(signal)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = cp.random.normal(0, cp.sqrt(noise_power), signal.shape)
    return signal + noise

# 并行化CFAR检测（GPU优化版）
def go_cfar_1d_gpu(signal, guard_cells, train_cells, alpha):
    N = len(signal)
    total_window = guard_cells + train_cells
    signal_ext = cp.pad(signal, (total_window, total_window), mode='constant')
    
    # 创建滑动窗口
    window_size = 2 * total_window + 1
    windows = cp.lib.stride_tricks.sliding_window_view(signal_ext, window_size)
    
    # 提取训练区并计算均值
    left = windows[:, :train_cells]
    right = windows[:, -train_cells:]
    train = cp.concatenate((left, right), axis=1)
    noise_est = cp.mean(train, axis=1)
    
    # 计算阈值并比较
    thresholds = alpha * noise_est
    central = windows[:, train_cells]  # 中心点
    cfar_mask = (central > thresholds).astype(int)
    
    # 去除padding部分
    return cfar_mask[total_window:-total_window], thresholds

# 蒙特卡洛试验函数（GPU版本）
def run_trial_gpu(trial_idx):
    cp.random.seed(trial_idx)
    random_num = cp.random.randint(0, 4000).item()
    
    # 生成发射信号
    Tx_paddle = cp.zeros(TIME*fs)
    start = random_num
    end = start + len(base_chirp)
    if end <= len(Tx_paddle):
        Tx_paddle[start:end] = base_chirp
    
    # 两次卷积
    s = csignal.fftconvolve(ir_gpu, Tx_paddle, mode='full')
    Rx = csignal.fftconvolve(ir_gpu, s, mode='full')[:len(Tx_paddle)]
    
    # 添加噪声和滤波
    Rx_noisy = awgn_gpu(Rx, args.snr)
    BPF_out = csignal.filtfilt(b_gpu, a_gpu, Rx_noisy)
    
    # 匹配滤波和包络检测
    matched = csignal.fftconvolve(BPF_out, template_gpu, mode='same')
    analytic = hilbert(matched.real)
    envelope = cp.abs(analytic)
    envelope /= cp.max(envelope)
    
    # CFAR检测
    cfar_mask, _ = go_cfar_1d_gpu(envelope, guard_len, train_len, alpha)
    detected = cp.where(cfar_mask == 1)[0]
    
    # 判断检测结果
    real_start = int(fs * min(ir['delay'])) + random_num
    real_end = int(fs * max(ir['delay'])) + random_num
    success = cp.any((detected >= real_start) & (detected <= real_end))
    return success.item()

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--snr', type=float, default=0)
    parser.add_argument('-o', '--output', default='results_gpu.txt')
    args = parser.parse_args()

    # 全局参数
    TIME = 200
    guard_len, train_len = 20, 400
    P_f = 1e-5
    alpha = 2 * (guard_len + train_len) * (P_f ** (-1/(2*(guard_len + train_len))) - 1)
    
    # 运行试验
    total_trials = 4000
    success_count = 0
    for _ in tqdm(range(total_trials), desc="GPU Trials"):
        success_count += run_trial_gpu(_)
    
    # 保存结果
    with open(args.output, 'a') as f:
        f.write("\n")
        f.write(f"虚警概率为: {P_f}\n")
        f.write(f"SNR: {args.snr}\n")
        f.write(f"蒙特卡洛实验次数: {total_trials}\n")
        f.write(f"检测成功次数: {success_count}\n")
        f.write(f"检测成功率: {success_count/total_trials:.2%}%\n")
        f.write("---------------------------------------------------------\n")

    print(f"结果已保存到 {args.output}")
duration = 0.5
f0, f1 = 1400, 1600
base_chirp = generate_chirp(duration, f0, f1, fs)
template_gpu = base_chirp[::-1].copy()  # 匹配滤波模板

# 噪声生成函数（GPU）
def awgn_gpu(signal, snr):
    signal_power = cp.sum(cp.abs(signal)**2) / len(signal)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = cp.random.normal(0, cp.sqrt(noise_power), signal.shape)
    return signal + noise

# 并行化CFAR检测（GPU优化版）
def go_cfar_1d_gpu(signal, guard_cells, train_cells, alpha):
    N = len(signal)
    total_window = guard_cells + train_cells
    signal_ext = cp.pad(signal, (total_window, total_window), mode='constant')
    
    # 创建滑动窗口
    window_size = 2 * total_window + 1
    windows = cp.lib.stride_tricks.sliding_window_view(signal_ext, window_size)
    
    # 提取训练区并计算均值
    left = windows[:, :train_cells]
    right = windows[:, -train_cells:]
    train = cp.concatenate((left, right), axis=1)
    noise_est = cp.mean(train, axis=1)
    
    # 计算阈值并比较
    thresholds = alpha * noise_est
    central = windows[:, train_cells]  # 中心点
    cfar_mask = (central > thresholds).astype(int)
    
    # 去除padding部分
    return cfar_mask[total_window:-total_window], thresholds

# 蒙特卡洛试验函数（GPU版本）
def run_trial_gpu(trial_idx):
    cp.random.seed(trial_idx)
    random_num = cp.random.randint(0, 4000).item()
    
    # 生成发射信号
    Tx_paddle = cp.zeros(TIME*fs)
    start = random_num
    end = start + len(base_chirp)
    if end <= len(Tx_paddle):
        Tx_paddle[start:end] = base_chirp
    
    # 两次卷积
    s = csignal.fftconvolve(ir_gpu, Tx_paddle, mode='full')
    Rx = csignal.fftconvolve(ir_gpu, s, mode='full')[:len(Tx_paddle)]
    
    # 添加噪声和滤波
    Rx_noisy = awgn_gpu(Rx, args.snr)
    BPF_out = csignal.filtfilt(b_gpu, a_gpu, Rx_noisy)
    
    # 匹配滤波和包络检测
    matched = csignal.fftconvolve(BPF_out, template_gpu, mode='same')
    analytic = hilbert(matched.real)
    envelope = cp.abs(analytic)
    envelope /= cp.max(envelope)
    
    # CFAR检测
    cfar_mask, _ = go_cfar_1d_gpu(envelope, guard_len, train_len, alpha)
    detected = cp.where(cfar_mask == 1)[0]
    
    # 判断检测结果
    real_start = int(fs * min(ir['delay'])) + random_num
    real_end = int(fs * max(ir['delay'])) + random_num
    success = cp.any((detected >= real_start) & (detected <= real_end))
    return success.item()

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--snr', type=float, default=0)
    parser.add_argument('-o', '--output', default='results_gpu.txt')
    args = parser.parse_args()

    # 全局参数
    TIME = 200
    guard_len, train_len = 20, 400
    P_f = 1e-5
    alpha = 2 * (guard_len + train_len) * (P_f ** (-1/(2*(guard_len + train_len))) - 1)
    
    # 运行试验
    total_trials = 4000
    success_count = 0
    for _ in tqdm(range(total_trials), desc="GPU Trials"):
        success_count += run_trial_gpu(_)
    
    # 保存结果
    with open(args.output, 'a') as f:
        f.write("\n")
        f.write(f"虚警概率为: {P_f}\n")
        f.write(f"SNR: {args.snr}\n")
        f.write(f"蒙特卡洛实验次数: {total_trials}\n")
        f.write(f"检测成功次数: {success_count}\n")
        f.write(f"检测成功率: {success_count/total_trials:.2%}%\n")
        f.write("---------------------------------------------------------\n")

    print(f"结果已保存到 {args.output}")