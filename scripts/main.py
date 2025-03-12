import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
#import librosa as lb
import arlpy.uwapm as pm
import arlpy.plot as aplt
# add/change bathy to env
import os
#uncomment this when using Macos ot linus
#os.environ["PATH"] += ":/Users/zanesing/Documents/at/Bellhop"
print(pm.models())
print(lb.__version__)
# 添加在文件开头的导入


#TODO SSP
# add/change SSP to env
ssp = [                                 #Munk 声速刨面
  [  0.0 , 1548.52],[200.0 , 1530.29],[400.0 , 1517.78],[600.0 , 1509.49 ],  
  [800.0 , 1504.30],[1000.0 , 1501.38],[1200.0 ,1500.14],[1400.0 ,  1500.12],  
  [1600.0 ,1501.02],[1800.0 , 1502.57] , [2000.0 , 1504.62],[2200.0 , 1507.02 ],  
  [2400.0 , 1509.69 ],[2600.0 , 1512.55],[2800.0 , 1515.56],[3000.0 , 1518.67 ],
  [ 3200.0 , 1521.85 ],[3400.0 , 1525.10 ],[3600.0 , 1528.38],[3800.0 , 1531.70],
  [4000.0 , 1535.04],[4200.0 , 1538.39],[4400.0 , 1541.76 ],[4600.0 , 1545.14],
  [4800.0 , 1548.52 ],[5000.0 , 1551.91]
]

# Appending ssp and bathy to existing env file
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

from scipy.signal import chirp
# 生成单个金字塔Chirp的函数
def generate_pyramid_chirp(duration, f0, f1, fs):
    """生成单个对称金字塔形Chirp信号"""
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    t_mid = duration / 2
    
    # 上升段
    up_segment = chirp(t[t <= t_mid], f0=f0, f1=f1, t1=t_mid, method='linear')
    # 下降段
    down_segment = chirp(t[t > t_mid] - t_mid, f0=f1, f1=f0, t1=t_mid, method='linear')
    
    return np.concatenate([up_segment, down_segment])

def ca_cfar_1d(signal, guard_cells=2, train_cells=4, alpha=5):
    """
    简易1D CA-CFAR实现:
    - signal: 输入1D数据
    - guard_cells: 保护单元数(单侧)
    - train_cells: 训练单元数(单侧)
    - alpha: 阈值放大系数(简化)
    返回: cfar_mask(0/1), 表示该点是否被判定为目标
    """
    N = len(signal)
    cfar_mask = np.zeros(N, dtype=int)
    # 滑窗边界
    total_window = guard_cells + train_cells
    
    for i in range(total_window, N - total_window):
        # 训练区提取 (不包含保护单元和测试单元)
        start = i - total_window
        end = i + total_window + 1
        
        # 左侧训练区: [start, i - guard_cells)
        # 右侧训练区: (i + guard_cells, end)
        left_train = np.abs(signal[start : i - guard_cells])
        right_train = np.abs(signal[i + guard_cells + 1 : end])
        
        train_zone = np.concatenate((left_train, right_train))
        noise_est = np.mean(train_zone)  # CA: 取平均
        
        threshold = alpha * noise_est
        if signal[i] > threshold:
            cfar_mask[i] = 1
    
    return cfar_mask

if __name__ == '__main__':

    #TODO: 参数配置
    fs = 4000           # 采样率
    c  = 1500           # 声速
    TIME = 200          # 信道选取时间

    #TODO: 计算声线传播
    env['rx_range']=500000   

    rays = pm.compute_rays(env , debug=False)
    pm.plot_rays(rays, env=env,width=900)

    #TODO: 根据arrival计算脉冲响应
    env['rx_range']=60000   #set target to 60km 
    arrivals = pm.compute_arrivals(env)

    #终端表格输出
    print("\n声波到达参数 (前10个路径):")
    print(tabulate(
        arrivals[arrivals.arrival_number < 10][['time_of_arrival', 'angle_of_arrival', 'surface_bounces', 'bottom_bounces']],
        headers=['到达时间(s)', '到达角度(度)', '海面反射', '海底反射'],
        tablefmt='github',
        floatfmt=(".3f", ".1f", ".0f", ".0f")
    ))
    #print(arrivals[arrivals.arrival_number < 10][['time_of_arrival', 'angle_of_arrival', 'surface_bounces', 'bottom_bounces']])
    ir = pm.arrivals_to_impulse_response(arrivals, fs=fs,abs_time=True)
    ir = np.concatenate([ir,np.zeros(int(TIME*fs))])


    """绘制单位脉冲响应"""
    t = np.arange(len(ir)) / fs
    IR_TIME = len(ir)/fs
    plt.figure()
    plt.plot(t, np.abs(ir))
    plt.xlabel('Time (s)');plt.ylabel('Amplitude');plt.title('Impulse Response')
    plt.grid(True);

    #TODO: 生成上下扫频（优化：预先生成避免重复计算）
    duration = 0.5      # 单个Chirp持续时间
    f0, f1 = 1400, 1600 # 频率范围
    base_chirp = generate_pyramid_chirp(duration, f0, f1, fs)
    template = base_chirp.copy()  # 上下扫频信号作为参考信号
    
    #高效拼接多个Chirp（避免使用循环内concatenate）
    num_chirp = 128      # 扫频信号重复次数
    Tx = np.tile(base_chirp, num_chirp)  # 使用np.tile代替循环

    #延长后补零
    Tx_paddle = np.concatenate([Tx,np.zeros(len(ir)-len(Tx))])  #长度等于相响应长度
    #绘制补零后的信号
    t = np.linspace(0, len(Tx_paddle) / fs, len(Tx_paddle))
    plt.figure()
    plt.title('transmitted signal')
    plt.plot(t,Tx_paddle)
    plt.grid(True);plt.ylim(-2,2);


    from scipy.signal import convolve


    #TODO: 两次卷积计算回波信号
    s = convolve( ir, Tx_paddle,mode='full')
    Rx = convolve(ir, s,mode='full')
        # You can optionally check the length of the results
    print("Length of transmitted signal: ", len(Tx_paddle))
    print("Length of first convolution result: ", len(s))
    print("Length of second convolution result: ", len(Rx))
    Rx = Rx[0:len(Tx_paddle)]

    #绘制
    plt.figure()
    plt.title('received signal')
    plt.plot(t, Rx)
    plt.grid(True)

    """加入噪声"""
    from modules.noise import awgn
    Rx_noisy = awgn(Rx, snr=10, out='signal', method='vectorized', axis=0)

    #TODO: 经过带通滤波器 , 匹配滤波器 再包络检波
    from scipy import signal  
    # 带通滤波器： 1350-1650Hz
    fmax = fs/2
    f_start ,f_end= 1350/fmax ,1650/fmax
    b, a  =   signal.butter( 8 , [ f_start , f_end ],  'bandpass' )    #配置滤波器 8 表示滤波器的阶数
    BPF_out  =   signal.filtfilt(b, a, Rx_noisy)   #data为要过滤的信号

    # 匹配滤波：将输入信号与参考信号的时间反转版本进行卷积
    matched_out = signal.fftconvolve(BPF_out, template[::-1], mode='same')

    # 包络检波: python中的hilbert是直接生成解析信号,不是hilbert变换 
    analytic = signal.hilbert(np.real(matched_out), N=None, axis=-1)
    envelope = np.abs(analytic)
    
    #绘制信号波形
    plt.figure(figsize=(10, 6))

    # 带通滤波信号
    plt.subplot(3, 1, 1)
    plt.plot(t, BPF_out);
    plt.title('Bandpass Output');plt.grid(True)
    # 匹配滤波信号
    plt.subplot(3, 1, 2)
    plt.plot(t, matched_out)
    plt.xlabel('Time (s)');plt.title('Matched Output')
    plt.grid(True)
    #包络检波后的信号
    plt.subplot(3, 1, 3)
    plt.plot(t, envelope)
    plt.xlabel('Time (s)');plt.title('Envelope')
    plt.grid(True)
    plt.tight_layout()

    #TODO: 回波信号进行1D-CA-CFAR

    guard_len ,train_len = 30,300  #设置守护单元和训练单元的长度
    alpha = 2.5                    #设置门限因子  
    N = guard_len+train_len    
    P_f = 1/((1 + alpha/N)**N)     # 计算设置的虚警概率
    
    cfar_mask = ca_cfar_1d(envelope, guard_cells=guard_len, train_cells=train_len, alpha=alpha)

    detected_indices = np.where(cfar_mask == 1)[0]

    # 绘制回波信号和检测结果
    plt.figure(figsize=(12, 6))
    plt.plot(t, envelope, label='Received Signal')
    plt.plot(t[detected_indices], np.abs(envelope[detected_indices]), 'rx', label='CFAR Detections')
    plt.title('CFAR Detection Results')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)


    plt.show()