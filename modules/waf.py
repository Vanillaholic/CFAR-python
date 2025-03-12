import numpy as np
from scipy.signal import resample, correlate
from scipy import signal
import matplotlib.pyplot as plt
from fractions import Fraction
import cupy as cp

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



def match_filter(data, signal, mode='signal'):
    """
    匹配滤波 (Matched Filtering)
    
    参数:
      data: 数据信号 (1D numpy 数组)
      signal: 模板信号 (1D numpy 数组)
      mode: 归一化类型 ('none', 'window', 'total', 'signal', 'sign')
    
    返回:
      xco: 互相关结果（非负时延部分）
    """
    if len(data) < len(signal):
        raise ValueError("Template signal is longer than the data")
    
    # 计算互相关（全长），注意 np.correlate 与 MATLAB xcorr 的实现略有不同，
    # 这里取非负时延部分，即从索引 len(data)-1 开始
    xco = correlate(data, signal, mode='full')
    xco = xco[len(data)-1:]
    
    # 归一化
    if mode == 'total':
        xco /= (np.sqrt(np.sum(data**2)) + np.finfo(float).eps)
        xco /= (np.sqrt(np.sum(signal**2)) + np.finfo(float).eps)
    elif mode == 'signal':
        xco /= (np.sum(signal**2) + np.finfo(float).eps)
    elif mode == 'window':
        window_energy = np.convolve(data**2, np.ones(len(signal)), mode='full')[len(data)-1:]
        xco /= np.sqrt(window_energy + np.finfo(float).eps)
        xco /= (np.sqrt(np.sum(signal**2)) + np.finfo(float).eps)
    elif mode == 'sign':
        xco /= np.sum(np.abs(signal))
    elif mode != 'none':
        print("*** Warning: unknown mode, defaulting to 'none'")
    
    return xco

def wideband_ambiguity(tx_signal, rx_signal, r, b, length, c, max_velocity, target_distance, fs):
    """
    计算宽带模糊函数（Wideband Ambiguity Function, WAF），并返回速度轴和距离轴
    
    参数:
      tx_signal: 发送信号 (numpy 数组)
      rx_signal: 接收信号 (numpy 数组)
      r: 雷达/声纳信号的带宽比例
      b: 频率伸缩因子
      length: 信号阶数
      c: 波速 (m/s)
      max_velocity: 最大速度范围 (m/s)
      target_distance: 目标距离 (m)
      fs: 采样频率 (Hz)
      
    返回:
      resampled_ambiguity: 模糊函数矩阵（降采样后），行对应时延，列对应速度
      velocity_axis: 速度轴 (m/s)
      range_axis: 距离轴 (m)
    """
    # 计算目标距离对应的时延（采样点数）
    num_samples = len(tx_signal)
    samples_delay = int(round((target_distance / c) * fs))
    total_obs_samples = samples_delay + num_samples

    # 生成延迟信号：先在 tx_signal 后补 zeros，再截取前 num_samples（等于 tx_signal），然后在前面补 samples_delay 个零
    extended_signal = np.concatenate([tx_signal, np.zeros(samples_delay)])
    truncated_signal = extended_signal[:num_samples]
    delayed_signal = np.concatenate([np.zeros(samples_delay), truncated_signal])
    
    # 计算速度分辨率
    velocity_resolution = c / (r * (b**length - 1))
    velocity_axis = np.arange(0, max_velocity + velocity_resolution, velocity_resolution)
    # 构造对称速度轴（负速度部分取正部分的反序，并去掉重复零）
    velocity_axis = np.concatenate([-velocity_axis[::-1][:-1], velocity_axis])
    
    # 计算多普勒伸缩因子
    doppler_factor = 1 + (velocity_axis / c)
    pq_list = []
    for df in doppler_factor:
        f = Fraction(df).limit_denominator()
        # p, q 和 df 的对应需要注意是否和 resample_poly 一致
        pq_list.append((f.denominator, f.numerator))  # (p, q) = (denominator, numerator)
    # 初始化模糊函数矩阵（长度与 delayed_signal 相同）
    ambiguity_matrix = np.zeros((len(delayed_signal), len(velocity_axis)))
    # Doppler scaling for each velocity
    for i, (vel, (p, q)) in enumerate(zip(velocity_axis, pq_list)):
        # doppler_scaled_signal = resample_poly(rx_signal, q, p)
        # 先注意：函数 resample_poly(x, up, down) => up = q, down = p
        doppler_scaled_signal = signal.resample_poly(rx_signal, q, p)

        # Zero padding to match signal length
        if len(doppler_scaled_signal) > len(delayed_signal):
            zero_padding = len(doppler_scaled_signal) - len(delayed_signal)
            delayed_signal_copy = np.concatenate([delayed_signal, np.zeros(zero_padding)])
        else:
            delayed_signal_copy = delayed_signal[:]

        if len(doppler_scaled_signal) < len(delayed_signal_copy):
            zero_padding = len(delayed_signal_copy) - len(doppler_scaled_signal)
            doppler_scaled_signal = np.concatenate([doppler_scaled_signal, np.zeros(zero_padding)])

        # Matched filter response
        mf_output = match_filter(delayed_signal_copy, doppler_scaled_signal, 'none')
        ambiguity_matrix[:, i] = np.abs(mf_output[:])  # store column

 
    # 首先确定行方向降采样后的新行数
    # resample_poly(…, 1, 6) => up=1, down=6 => 行数/6 (大约)
    original_rows = ambiguity_matrix.shape[0]
    # 大约输出行数
    downsampled_rows = int(np.ceil(original_rows / 6))
    # 初始化 resampled_ambiguity
    resampled_ambiguity = np.zeros((downsampled_rows, ambiguity_matrix.shape[1]))

    for i in range(len(velocity_axis)):
        # 对 ambiguity_matrix(:, i) 做下采样
        col_resampled = signal.resample_poly(ambiguity_matrix[:, i], 1, 6)
        # col_resampled 的长度可能不是正好等于 downsampled_rows，需要截断或补零
        if len(col_resampled) > downsampled_rows:
            col_resampled = col_resampled[:downsampled_rows]
        elif len(col_resampled) < downsampled_rows:
            col_resampled = np.concatenate([col_resampled, np.zeros(downsampled_rows - len(col_resampled))])
        resampled_ambiguity[:, i] = col_resampled

    # 计算距离坐标轴
    num_delays = resampled_ambiguity.shape[0]
    delay_index = np.arange(0, num_delays)
    range_axis = (delay_index) * (c / (2 * fs))

    return resampled_ambiguity, velocity_axis, range_axis

if __name__ == "__main__":

    # 参数配置
    fs = 4000           # 采样率
    duration = 0.62      # 单个子信号
    f0 = 1500            # 中心频率

    subpulse1,template = generate_phase_encoded_signal(mseq1,duration,f0,fs)
    subpulse2,  _      = generate_phase_encoded_signal(mseq2,duration,f0,fs)



    WAF, velocity_axis, range_axis = wideband_ambiguity(cp.asnumpy(subpulse1), cp.asnumpy(subpulse1),
                                                                    r=50, b=2, length=7, c=1500,
                                                                    max_velocity=10, target_distance=1000, fs=fs)

    plt.figure(figsize=(8, 6))
    # 用 extent 指定 x 轴为 velocity，y 轴为 range（距离），这里 range_axis 的最大值对应图像的高度
    plt.imshow(WAF, aspect='auto', extent=[velocity_axis.min(), velocity_axis.max(), range_axis[-1], range_axis[0]])
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.title("Wideband Ambiguity Function")
    plt.colorbar() 
   

    from mpl_toolkits.mplot3d import Axes3D
    X, Y = np.meshgrid(velocity_axis, range_axis)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, WAF, cmap='viridis')

    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_zlabel("WAF Amplitude")
    ax.set_title("Wideband Ambiguity Function (3D Surface)")

    # 添加 colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.show()
