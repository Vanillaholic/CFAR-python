import numpy as np
import matplotlib.pyplot as plt
import arlpy.uwapm as pm
import arlpy.plot as aplt
from noise import awgn
import os
import argparse
from tqdm import tqdm
from scipy import signal
from scipy.signal import chirp, convolve, butter, filtfilt, fftconvolve, hilbert
from noise import awgn  # 请确保已安装 noise 包
from multiprocessing import Pool

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
# 生成单个Chirp的函数
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
    #TODO: 导入terminal参数
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--snr', type=float, default=0, help='set the signal-noise ratio')
parser.add_argument('-o','--output', type=str, default='results.txt', help='保存结果的文件名')
# 解析参数
args = parser.parse_args()

# #TODO: 参数配置
fs = 4000           # 采样率
c  = 1500           # 声速
TIME = 200          # 信道选取时间
snr = args.snr

# #TODO: 根据arrival计算脉冲响应
env['rx_range']=60000   #set target to 60km 
arrivals = pm.compute_arrivals(env)

ir = pm.arrivals_to_impulse_response(arrivals, fs=fs,abs_time=True)

first_realtime ,last_realtime= 2*np.min(arrivals['time_of_arrival']),2*np.max(arrivals['time_of_arrival'])
real_echo_start = int(fs*first_realtime)
real_echo_end = int(fs*last_realtime)

# #TODO: 生成扫频
duration = 0.5      # 单个Chirp持续时间
f0, f1 = 1400, 1600 # 频率范围
base_chirp = generate_chirp(duration, f0, f1, fs)
template = base_chirp.copy()  # 上下扫频信号作为参考信号
    
#设置Chirp次数（避免使用循环内concatenate）
num_chirp = 1      # 扫频信号重复次数
#Tx = np.tile(base_chirp, num_chirp)  # 使用np.tile代替循环
Tx = base_chirp

'''带通滤波参数'''
# 带通滤波器： 1350-1650Hz
fmax = fs/2
f_start ,f_end= 1350/fmax ,1650/fmax
b, a  =   signal.butter( 8 , [ f_start , f_end ],  'bandpass' )    #配置滤波器 8 表示滤波器的阶数

'''CFAR参数'''
guard_len ,train_len = 20,400  #设置守护单元和训练单元的长度
N = guard_len+train_len  
P_f = 1e-5                   # 计算设置的虚警概率
alpha = 2 * N * (np.power(P_f, -1 / (2 * N)) - 1)         #设置门限因子 


def run_trial(trial_idx):
    # 为保证不同试验产生不同随机数序列，可使用 trial_idx 作为随机种子
    np.random.seed(trial_idx)
    # 根据偏移量调整真实回波检测范围
    '''发射信号补零,random_num表示发射时刻不确定'''
    random_num = np.random.randint(0,4000)
    trial_real_echo_start = int(fs * first_realtime) + random_num
    trial_real_echo_end = int(fs * last_realtime) + random_num

    # 两次卷积计算回波信号
    Tx_paddle = np.concatenate([np.zeros(random_num),Tx,np.zeros(TIME*fs-len(Tx)-random_num)])  #长度等于相响应长度
    s = convolve(ir, Tx_paddle, mode='full')
    Rx = convolve(ir, s, mode='full')
    Rx = Rx[:len(Tx_paddle)]
    
    # 加入噪声
    Rx_noisy = awgn(Rx, snr=args.snr, out='signal', method='vectorized', axis=0)
    
    # 带通滤波（滤波器系数已预计算）
    BPF_out = filtfilt(b, a, Rx_noisy)
    
    # 匹配滤波（利用 FFT 加速卷积）
    matched_out = fftconvolve(BPF_out, template[::-1], mode='same')
    
    # 包络检测（Hilbert 变换）
    analytic = hilbert(np.real(matched_out))
    envelope = np.abs(analytic)
    envelope = envelope / np.max(envelope)
    
                 # 计算设置的虚警概率
    cfar_mask, thresholds = ca_cfar_1d(envelope, guard_cells=guard_len, train_cells=train_len, alpha=alpha)
    detected_indices = np.where(cfar_mask == 1)[0]
    
    # 判断是否在真实回波范围内检测到目标
    success = np.any((detected_indices >= trial_real_echo_start) & (detected_indices < trial_real_echo_end))
    return 1 if success else 0

if __name__ == "__main__":


    '''利用 multiprocessing.Pool 并行化 Monte Carlo 模拟'''
    success_count = 0
    total_epochs = 4000  # 总的 epoch 数量
    with Pool(processes=10) as pool:
         # 使用 tqdm 显示进度
        results = list(tqdm(pool.imap(run_trial, range(total_epochs)),
                             total=total_epochs, desc="Running Monte Carlo experiment"),unit="epoch")
    
    success_count = sum(results)

    # 保存结果到文件
    with open(args.output, 'a',encoding='utf-8') as f:
        f.write("\n")
        f.write(f"虚警概率为: {P_f}\n")
        f.write(f"守护单元{guard_len},训练单元{train_len}\n")
        f.write(f"SNR: {args.snr}\n")
        f.write(f"蒙特卡洛实验次数: {total_epochs}\n")
        f.write(f"检测成功次数: {success_count}\n")
        f.write(f"检测成功率: {success_count / total_epochs * 100:.2f}%\n")
        f.write("---------------------------------------------------------\n")

    print(f"结果已保存到 {args.output}")

