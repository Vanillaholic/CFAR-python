import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
#import librosa as lb
import arlpy.uwapm as pm
import arlpy.plot as aplt
# add/change bathy to env
import os
import argparse
from tqdm import tqdm
#uncomment this when using Macos ot linus
#os.environ["PATH"] += ":/Users/zanesing/Documents/at/Bellhop"
# print(pm.models())
# print(lb.__version__)
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
def generate_chirp(duration, f0, f1, fs):
    """生成单个Chirp信号"""
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    return chirp(t, f0=f0, f1=f1, t1=duration, method='linear')

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
    threshold_value = np.zeros(N)
    for i in range(total_window, N - total_window):

        # 训练区提取 (不包含保护单元和测试单元)
        start = i - total_window
        end = i + total_window + 1
        
        # 左侧训练区: [start, i - guard_cells)
        # 右侧训练区: (i + guard_cells, end)
        left_train = signal[start : i - guard_cells]
        right_train = signal[i + guard_cells + 1 : end]
        
        train_zone = np.concatenate((left_train, right_train))
        noise_est = np.mean(train_zone)  # CA: 取平均
        
        threshold = alpha * noise_est
        if signal[i] > threshold:
            cfar_mask[i] = 1
        threshold_value[i] = threshold
    return cfar_mask , threshold_value

if __name__ == '__main__':

    #TODO: 导入terminal参数
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--snr', type=float, default=-5, help='set the signal-noise ratio')
    parser.add_argument('-o', type=str, default='result.txt', help='保存结果的文件名')
    # 解析参数
    args = parser.parse_args()

    #TODO: 参数配置
    fs = 4000           # 采样率
    c  = 1500           # 声速
    TIME = 200          # 信道选取时间

    #TODO: 计算声线传播
    env['rx_range']=500000   

    rays = pm.compute_rays(env , debug=False)
    #pm.plot_rays(rays, env=env,width=900)

    #TODO: 根据arrival计算脉冲响应
    env['rx_range']=60000   #set target to 60km 
    arrivals = pm.compute_arrivals(env)

    #终端表格输出
    #print("\n声波到达参数 (前10个路径):")
    #print(tabulate(
    #    arrivals[arrivals.arrival_number < 10][['time_of_arrival', 'angle_of_arrival', 'surface_bounces', 'bottom_bounces']],
    #    headers=['到达时间(s)', '到达角度(度)', '海面反射', '海底反射'],
    #    tablefmt='github',
    #    floatfmt=(".3f", ".1f", ".0f", ".0f")
    #))
    #print(arrivals[arrivals.arrival_number < 10][['time_of_arrival', 'angle_of_arrival', 'surface_bounces', 'bottom_bounces']])
    ir = pm.arrivals_to_impulse_response(arrivals, fs=fs,abs_time=True)
    ir = np.concatenate([ir,np.zeros(int(TIME*fs))])

    first_realtime ,last_realtime= 2*np.min(arrivals['time_of_arrival']),2*np.max(arrivals['time_of_arrival'])
    real_echo_start = int(fs*first_realtime)
    real_echo_end = int(fs*last_realtime)

    """绘制单位脉冲响应"""
    t = np.arange(len(ir)) / fs
    IR_TIME = len(ir)/fs
    # plt.figure()
    # plt.plot(t, np.abs(ir))
    # plt.xlabel('Time (s)');plt.ylabel('Amplitude');plt.title('Impulse Response')
    # plt.grid(True);

    #TODO: 生成扫频
    duration = 0.5      # 单个Chirp持续时间
    f0, f1 = 1400, 1600 # 频率范围
    base_chirp = generate_chirp(duration, f0, f1, fs)
    template = base_chirp.copy()  # 上下扫频信号作为参考信号
    
    #设置Chirp次数（避免使用循环内concatenate）
    num_chirp = 1      # 扫频信号重复次数
    #Tx = np.tile(base_chirp, num_chirp)  # 使用np.tile代替循环
    Tx = base_chirp

    success_count = 0
    total_epochs = 8000  # 总的 epoch 数量
    # 使用 tqdm 包装循环
    for epoch in tqdm(range(1, int(total_epochs) + 1), desc="running CFAR experiment", unit="epoch"):
        '''发射信号补零,random_num表示发射时刻不确定'''
        random_num = np.random.randint(0,100)
        Tx_paddle = np.concatenate([np.zeros(random_num),Tx,np.zeros(len(ir)-len(Tx)-random_num)])  #长度等于相响应长度
        #绘制补零后的信号
        t = np.linspace(0, len(Tx_paddle) / fs, len(Tx_paddle))
        '''
        plt.figure()
        plt.title('transmitted signal')
        plt.plot(t,Tx_paddle)
        plt.grid(True);plt.ylim(-2,2);
        '''

        #TODO: 两次卷积计算回波信号
        from scipy.signal import convolve
        s = convolve( ir, Tx_paddle,mode='full')
        Rx = convolve(ir, s,mode='full')
        Rx = Rx[0:len(Tx_paddle)]

        # #绘制
        # plt.figure()
        # plt.title('received signal')
        # plt.plot(t, Rx)
        # plt.grid(True)

        """加入噪声"""
        from noise import awgn
        Rx_noisy = awgn(Rx, snr=args.snr, out='signal', method='vectorized', axis=0)

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
        '''
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
        plt.plot(t[int(79.75*fs):int(80.25*fs)], envelope[int(79.75*fs):int(80.25*fs)])
        plt.xlabel('Time (s)');plt.title('Envelope')
        plt.grid(True)
        plt.tight_layout()
        '''
        envelope = envelope/np.max(envelope)

        #TODO: 回波信号进行1D-CA-CFAR

        guard_len ,train_len = 20,500  #设置守护单元和训练单元的长度
        N = guard_len+train_len  
        #alpha=  5
        #P_f= (alpha/(2*N)+1)**(-2*N)
        P_f = 5e-5                   # 计算设置的虚警概率
        alpha = 2 * N * (np.power(P_f, -1 / (2 * N)) - 1)         #设置门限因子 
        '''
        print("虚警概率为:",P_f)
        '''
        cfar_mask , thresholds = ca_cfar_1d(envelope, guard_cells=guard_len, train_cells=train_len, alpha=alpha)
        detected_indices = np.where(cfar_mask == 1)[0]

        success = np.any((detected_indices >= real_echo_start) & (detected_indices < real_echo_end))
        if success:
            #在真实回波范围内检测到信号，初步判定检测成功
            success_count += 1
        #在真实回波范围内未检测到信号，检测失败

        # # 绘制回波信号和检测结果
        # plt.figure(figsize=(12, 6))
        # plt.subplot(2,1,1)
        # plt.plot(t, envelope, label='Received Signal')
        # plt.plot(t,thresholds,label='Threshold')
        # plt.plot(t[detected_indices], envelope[detected_indices], 'rx', label='CFAR Detections')
        # plt.title(f'CFAR Detection Results($P_f=${P_f})')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.legend()
        # plt.grid(True)

        # # Zoom the result
        # buffer = envelope[int(79.75*fs):int(80.25*fs)]
        # t_buffer = t[int(79.75*fs):int(80.25*fs)]
        # thresholds_buffer = thresholds[int(79.75*fs):int(80.25*fs)]
        # cfar_mask_buffer = cfar_mask[int(79.75*fs):int(80.25*fs)]
        # detected_indices = np.where(cfar_mask_buffer == 1)[0]

        # # 绘制放大结果
        # plt.subplot(2,1,2)
        # plt.plot(t_buffer, buffer, label='Buffer')
        # plt.plot(t_buffer,thresholds_buffer,label='Threshold')
        # plt.plot(t_buffer[detected_indices], buffer[detected_indices], 'rx', label='CFAR Detections')
        # plt.title('Zoomed Results')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()

        # plt.show()
        
    # 保存结果到文件
    with open(args.output_file, 'w') as f:
        f.write(f"虚警概率为: {P_f}\n")
        f.write(f"SNR: {args.snr}\n")
        f.write(f"蒙特卡洛实验次数: {total_epochs}\n")
        f.write(f"检测成功次数: {success_count}\n")
        f.write(f"检测成功率: {success_count / total_epochs * 100:.2f}%\n")

    print(f"结果已保存到 {args.output_file}")

