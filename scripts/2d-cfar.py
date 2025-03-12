import numpy as np

def range_fft(IF_mat, Nr):
    """
    对每个 Chirp（列）执行范围 FFT。
    :param IF_mat: 输入的二维信号矩阵，形状 (Nr, Nd)
    :param Nr:     每个 Chirp 的采样点数（快时间维度）
    :return:       经过范围 FFT 变换后的矩阵，形状仍为 (Nr, Nd)
    """
    range_win = cp.hamming(Nr)  # 生成范围窗
    Nd = IF_mat.shape[1]        # 列数（Chirp 数）

    for i in range(Nd):
        # 取第 i 个 Chirp 的快时间数据（这一列）
        temp = IF_mat[:, i] * range_win
        # 对快时间方向做 Nr 点 FFT
        IF_mat[:, i] = cp.fft.fft(temp, Nr)
    return IF_mat

def doppler_fft(IF_mat, Nd):
    """
    对每个 Range Bin（行）执行 Doppler FFT。
    :param IF_mat: 经过范围 FFT 变换后的矩阵，形状 (Nr, Nd)
    :param Nd:     Chirp 个数（慢时间采样点数）
    :return:       经过 Doppler FFT 变换后的矩阵，形状仍为 (Nr, Nd)
    """
    doppler_win = cp.hamming(Nd)  # 生成 Doppler 窗
    Nr = IF_mat.shape[0]          # 行数（Range Bin 数）

    for j in range(Nr):
        # 取第 j 个 Range Bin 的所有 Chirp 数据（这一行）
        temp = IF_mat[j, :] * doppler_win
        # 对慢时间方向做 Nd 点 FFT，并使用 fftshift 将零频移到中心
        IF_mat[j, :] = cp.fft.fftshift(cp.fft.fft(temp, Nd))
    return IF_mat

#TODO 截取一段回波信号(假设这段是回波)
echo_start_idx = int(2*np.min(arrivals['time_of_arrival'])*fs)
echo_end_idx =  echo_start_idx + len(Tx) 
echo = Rx[echo_start_idx:echo_end_idx]  

IF = Tx*cp.conj(echo)

Nr = int(duration*fs)
Nd = int(128)
IF_mat = IF.reshape(Nr,Nd)

B = 200
c = 1500
fc = 1500

'''计算range-doppler  map'''
ranged_mat = range_fft(IF_mat.copy(),Nr)
range_doppler = doppler_fft(ranged_mat.copy(),Nd)
#取其幅度
RDM = np.abs(range_doppler).T
RDM /= np.max(RDM)

'''计算距离轴'''
max_range = (  c  * fs * duration / B/ 2)# 计算最大探测距离: c*Nr /2B
range_axis = cp.linspace(0, max_range, Nr, endpoint=False) 
range_axis = range_axis + cp.asarray(np.min(arrivals['time_of_arrival'])*c)

'''计算速度轴'''
doppler_axis = cp.linspace(-1 / duration / 2,1 / duration / 2,Nd,endpoint=False)#doppler axis:-1/(2T)~1/(2T). T位脉冲周期
velocity_axis = doppler_axis*c/fc/2      #fdc/(2fc)

# 绘图
plt.figure()
plt.imshow(RDM,
           aspect='auto', 
           cmap='jet',
           extent=[range_axis[0], range_axis[-1], velocity_axis[-1], velocity_axis[0]]) 
plt.title('Range Doppler Map')
plt.xlabel('range (m)')
plt.ylabel('velocity (m/s)')
plt.colorbar()
plt.tight_layout()