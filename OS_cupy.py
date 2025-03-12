import cupy as cp
import cupyx.scipy.signal as csignal
from cupyx.scipy.signal import hilbert
import arlpy.uwapm as pm
import numpy as np
from scipy.signal import butter
from scipy import signal
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
duration = 0.1

#f0, f1 = 1400, 1600
# base_chirp = generate_chirp(duration, f0, f1, fs)
# template_gpu = base_chirp[::-1].copy()  # 匹配滤波模板

# 定义生成相位编码信号的函数
def generate_phase_encoded_signal(m_sequence, duration, f0, fs):
    t = cp.linspace(0, duration, int(duration * fs), endpoint=False)  # 时间轴
    # 初始化相位编码信号
    subpulse = cp.zeros_like(t)
    # 遍历 m-sequence，根据其值调整相位
    for i, bit in enumerate(m_sequence):
        start = int(i * len(t) / len(m_sequence))
        end = int((i + 1) * len(t) / len(m_sequence))
        if bit == 1:
            subpulse[start:end] = cp.cos(2 * cp.pi * f0 * t[start:end] + cp.pi)
        else:
            subpulse[start:end] = cp.cos(2 * cp.pi * f0 * t[start:end])
    template = subpulse[::-1]
    return subpulse , template
mseq1 = cp.asarray([0,0,0,0,1,0,0,1,0,1,1,0,0,1,1,1,1,1,0,0,0,1,1,0,1,1,1,0,1,0,1])
mseq2 = cp.asarray([0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1])
mseq3 = cp.asarray([0,0,0,0,1,1,0,0,1,0,0,1,1,1,1,1,0,1,1,1,0,0,0,1,0,1,0,1,1,0,1])
f0 = 1500
subpulse1,template = generate_phase_encoded_signal(mseq1,duration,f0,fs)
subpulse2,  _      = generate_phase_encoded_signal(mseq2,duration,f0,fs)
subpulse3,  _      = generate_phase_encoded_signal(mseq3,duration,f0,fs)
pulse = cp.concatenate([subpulse1,subpulse2,subpulse3])
template_gpu = subpulse1[::-1].copy()  # 匹配滤波模板


# 噪声生成函数（GPU）
def awgn_gpu(signal, snr):
    signal_power = cp.sum(cp.abs(signal)**2) / len(signal)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = cp.random.normal(0, cp.sqrt(noise_power), signal.shape)
    return signal + noise

'''并行化CFAR检测（GPU优化版）'''
def ca_cfar_1d_gpu(signal, guard_cells, train_cells, alpha):
    '''
    CA-CFAR算法
    '''
    N = len(signal)
    total_window = guard_cells + train_cells
    signal_ext = cp.pad(signal, (total_window, total_window), mode='constant')
    #信号的长度增加了 2 * total_window，
    #这样可以确保滑动窗口操作后的输出与输入信号的大小一致。
    
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
    central = windows[:, train_cells+guard_cells]  # 中心点
    cfar_mask = (central > thresholds).astype(int)
    
    # 去除padding部分
    return cfar_mask[total_window:-total_window], thresholds
def go_cfar_1d_gpu(signal, guard_cells, train_cells, alpha):
    '''
    GO-CFAR算法
    '''
    N = len(signal)
    total_window = guard_cells + train_cells
    signal_ext = cp.pad(signal, (total_window, total_window), mode='constant')
    
    # 创建滑动窗口
    window_size = 2 * total_window + 1
    windows = cp.lib.stride_tricks.sliding_window_view(signal_ext, window_size)
    
    # 提取训练区并计算均值
    left = windows[:, :train_cells]
    right = windows[:, -train_cells:]
    mean1 = cp.mean(left,axis=1)
    mean2 = cp.mean(right,axis=1)
    
    # 逐元素比较，选择较大的均值作为噪声估计
    noise_est = cp.maximum(mean1, mean2)
    
    # 计算阈值并比较
    thresholds = alpha * noise_est
    central = windows[:, train_cells+guard_cells]  # 中心点
    cfar_mask = (central > thresholds).astype(int)
    
    # 去除padding部分
    return cfar_mask[total_window:-total_window], thresholds

def so_cfar_1d_gpu(signal, guard_cells, train_cells, alpha):
    '''
    SO-CFAR算法
    '''
    N = len(signal)
    total_window = guard_cells + train_cells
    signal_ext = cp.pad(signal, (total_window, total_window), mode='constant')
    
    # 创建滑动窗口
    window_size = 2 * total_window + 1
    windows = cp.lib.stride_tricks.sliding_window_view(signal_ext, window_size)
    
    # 提取训练区并计算均值
    left = windows[:, :train_cells]
    right = windows[:, -train_cells:]
    mean1 = cp.mean(left,axis=1)
    mean2 = cp.mean(right,axis=1)
    
    # 逐元素比较，选择较小的均值作为噪声估计
    noise_est = cp.minimum(mean1, mean2)
    
    # 计算阈值并比较
    thresholds = alpha * noise_est
    central = windows[:, train_cells+guard_cells]  # 中心点
    cfar_mask = (central > thresholds).astype(int)
    
    # 去除padding部分
    return cfar_mask[total_window:-total_window], thresholds


def os_cfar_1d_gpu(signal, guard_cells, train_cells, alpha, k):
    '''
    OS-CFAR算法
    :param signal: 输入信号
    :param guard_cells: 保护单元数
    :param train_cells: 训练单元数
    :param alpha: 阈值乘子
    :param k: 排序后选择的第k个值（从0开始索引）
    :return: CFAR掩码和阈值
    '''
    N = len(signal)
    total_window = guard_cells + train_cells
    signal_ext = cp.pad(signal, (total_window, total_window), mode='constant')
    
    # 创建滑动窗口
    window_size = 2 * total_window + 1
    windows = cp.lib.stride_tricks.sliding_window_view(signal_ext, window_size)
    
    # 提取训练区（左侧和右侧）
    left = windows[:, :train_cells]
    right = windows[:, -train_cells:]
    train_region = cp.concatenate((left, right), axis=1)  # 合并训练区
    
    # 对训练区进行排序
    sorted_train_region = cp.sort(train_region, axis=1)
    
    # 选择第k个值作为噪声估计
    noise_est = sorted_train_region[:, k]
    
    # 计算阈值并比较
    thresholds = alpha * noise_est
    central = windows[:, train_cells+guard_cells]  # 中心点
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
    end = start + len(pulse)
    if end <= len(Tx_paddle):
        Tx_paddle[start:end] = pulse
    
    # 两次卷积
    # s = csignal.fftconvolve(ir_gpu, Tx_paddle, mode='full')
    # Rx = csignal.fftconvolve(ir_gpu, s, mode='full')[:len(Tx_paddle)]
    s = signal.fftconvolve(ir_gpu.get(), Tx_paddle.get(), mode='full')
    Rx = signal.fftconvolve(ir_gpu.get(), s, mode='full')[:len(Tx_paddle)]
    Rx = cp.asarray(Rx)
    # 添加噪声和滤波
    Rx_noisy = awgn_gpu(Rx, args.snr)
    BPF_out = csignal.filtfilt(b_gpu, a_gpu, Rx_noisy)
    
    # 匹配滤波和包络检测
    matched = csignal.fftconvolve(BPF_out, template_gpu, mode='same')
    analytic = hilbert(matched.real)
    envelope = cp.abs(analytic)
    #envelope /= cp.max(envelope)

    
    #降采样 ，采样率为原来的1/10
    q = 100
    envelope_res = csignal.decimate(envelope,q, n=None, ftype='iir', axis=-1, zero_phase=True) 
    fs_new = fs / q 
    #t_res = cp.arange(0, len(envelope_res)) / fs_new 
    
    # CFAR检测
    cfar_mask, _ = os_cfar_1d_gpu(envelope_res, guard_len, train_len, alpha,k=10)
    detected = cp.where(cfar_mask == 1)[0]+ train_len+guard_len
    
    # 判断检测结果
    real_start = 2*int(fs_new * cp.min(arrivals['time_of_arrival']) )+ random_num//q
    real_end = 2*int(fs_new * cp.max(arrivals['time_of_arrival'])) + random_num//q
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
    guard_len, train_len = 10, 25
    P_f = 1e-5
    alpha = 2 * (guard_len + train_len) * (P_f ** (-1/(2*(guard_len + train_len))) - 1)
    
    # 运行试验
    total_trials = 10
    success_count = 0
    for _ in tqdm(range(total_trials), desc="GPU Trials"):
        success_count += run_trial_gpu(_)
    
    # 保存结果
    with open(args.output, 'a',encoding='utf-8') as f:
        f.write("\n")
        f.write(f"虚警概率为: {P_f}\n")
        f.write(f"守护单元{guard_len},训练单元{train_len}\n")
        f.write(f"SNR: {args.snr}\n")
        f.write(f"蒙特卡洛实验次数: {total_trials}\n")
        f.write(f"检测成功次数: {success_count}\n")
        f.write(f"检测成功率: {success_count/total_trials:.2%}%\n")
        f.write("---------------------------------------------------------\n")

    print(f"结果已保存到 {args.output}")