import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb, gamma

# ============== 1D CA-CFAR 示例 ==============
def ca_cfar_1d(signal, guard_cells=2, train_cells=4, alpha=5):
    """
    简易1D CA-CFAR实现:
    - signal: 输入1D数据
    - guard_cells: 保护单元数(单侧)
    - train_cells: 训练单元数(单侧)
    - alpha: 阈值放大系数(简化)
    返回: cfar_mask(0/1), 表示该点是否被判定为目标,threshold_value,阈值变化
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

def go_cfar_1d(signal, guard_cells=2, train_cells=4, alpha=5):
    """
    简易1D GO-CFAR实现(最大):
    使用时候注意取模长
    - signal: 输入1D数据
    - guard_cells: 保护单元数(单侧)
    - train_cells: 训练单元数(单侧)
    - alpha: 阈值放大系数(简化)
    返回: cfar_mask(0/1), 表示该点是否被判定为目标,threshold_value,阈值变化
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
        
        buffer1 = np.mean(left_train)
        buffer2 = np.mean(right_train)
        if buffer1 > buffer2 :  #均值进行比较取最大值
            noise_est = buffer1
        else:
            noise_est = buffer2
        
        threshold = alpha * noise_est
        if signal[i] > threshold:
            cfar_mask[i] = 1
        threshold_value[i] = threshold
    return cfar_mask , threshold_value


def so_cfar_1d(signal, guard_cells=2, train_cells=4, alpha=5 ):
    """
    简易1D SO-CFAR实现(最小):
    使用时候注意取模长
    - signal: 输入1D数据
    - guard_cells: 保护单元数(单侧)
    - train_cells: 训练单元数(单侧)
    - alpha: 阈值放大系数(简化)
    返回: cfar_mask(0/1), 表示该点是否被判定为目标,threshold_value,阈值变化
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
        
        buffer1 = np.mean(left_train)
        buffer2 = np.mean(right_train)
        if buffer1 < buffer2 :  #均值进行比较取最小值
            noise_est = buffer1
        else:
            noise_est = buffer2
        
        threshold = alpha * noise_est
        if signal[i] > threshold:
            cfar_mask[i] = 1
        threshold_value[i] = threshold
    return cfar_mask , threshold_value

def os_cfar_1d(signal, guard_cells=2, train_cells=4, alpha=5 , k=5, T=0.5):
    """
    简易1D SO-CFAR实现(有序统计):
    使用时候注意取模长
    - signal: 输入1D数据
    - guard_cells: 保护单元数(单侧)
    - train_cells: 训练单元数(单侧)
    - alpha: 阈值放大系数(简化)
    - 调整噪声估计的灵敏度：
    当 T=0 时，公式简化为标准的有序统计CFAR（OS-CFAR）的噪声估计。
    当 T>0 时，噪声估计会变得更加保守，从而降低虚警概率，但可能会增加漏检概率。
    当 T<0 时，噪声估计会变得更加敏感，从而增加虚警概率，但可能会减少漏检概率。
    返回: cfar_mask(0/1), 表示该点是否被判定为目标,threshold_value,阈值变化
    """
    N = len(signal)
    cfar_mask = np.zeros(N, dtype=int)
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
        
        # 对训练区进行排序
        sorted_train = np.sort(train_zone)
        
        # 选取第k小的值作为噪声估计的基础
        kth_value = sorted_train[k - 1]  # 因为索引从0开始

        # 计算公式中的参数
        R = len(train_zone)  # 训练单元总数
        comb_R_k = comb(R, k, exact=False)  # 组合数 C(R, k)
        gamma_R_k_T = gamma(R - k + 1 + T)  # Γ(R - k + 1 + T)
        gamma_k = gamma(k)  # Γ(k)
        gamma_R_T = gamma(R + 1 + T)  # Γ(R + 1 + T)

        # 计算 P_{fa, os}
        P_fa_os = k * comb_R_k * (gamma_R_k_T * gamma_k) / gamma_R_T

        # 噪声估计
        noise_est = kth_value * P_fa_os

        # 计算阈值
        threshold = alpha * noise_est

        # 判断是否为目标
        if signal[i] > threshold:
            cfar_mask[i] = 1

        # 保存阈值
        threshold_value[i] = threshold

    return cfar_mask, threshold_value

import numpy as np

def generate_RDM_data(St, Sr, M):
    """
    生成线性调频（chirp）信号的中频数据

    参数:
        St (numpy.ndarray): 一维数组，表示发射信号
        Sr (numpy.ndarray): 一维数组，表示回波信号
        M (int): chirp信号的数目

    返回:
        IF_mat (numpy.ndarray): 中频信号矩阵，形状为(M, len(St)) 横轴为距离信息,纵轴为多普勒信息
    """
    # 检查输入信号的长度是否一致
    if len(St) != len(Sr):
        raise ValueError("发射信号和回波信号的长度必须一致！")

    N = len(St)  # 信号长度
    IF_mat = np.zeros((M, N), dtype=complex)  # 初始化中频信号矩阵

    # 循环生成每个chirp的中频信号
    for i in range(M):
        Sr_conj = np.conj(Sr)  # 回波信号的共轭
        IF = St * Sr_conj  # 中频信号
        IF_mat[i, :] = IF  # 保存中频信号
    # 对中频信号矩阵进行二维傅里叶变换 (2D FFT)
    RDM = np.fft.fft2(IF_mat)  # 2D FFT
    RDM = np.fft.fftshift(RDM, axes=(0, 1))  # 将零频移到矩阵中心

    return RDM



def ca_cfar_2d(matrix, guard_cells=1, train_cells=2, alpha=5):
    """
    简易2D CA-CFAR (在距离-多普勒平面)
    - matrix: 输入2D矩阵 (距离 x 多普勒)
    - guard_cells: 每个方向上的保护单元大小
    - train_cells: 每个方向上的训练单元大小
    - alpha: 阈值放大系数(简化)
    返回: cfar_mask(同样大小的2D array), 1表示检测到目标, 0表示噪声
    """
    nr, nd = matrix.shape
    cfar_mask = np.zeros((nr, nd), dtype=int)
    
    # 逐个单元滑窗处理
    for r in range(train_cells+guard_cells, nr - train_cells - guard_cells):
        for d in range(train_cells+guard_cells, nd - train_cells - guard_cells):
            
            # 取训练区
            r_start = r - train_cells - guard_cells
            r_end   = r + train_cells + guard_cells + 1
            d_start = d - train_cells - guard_cells
            d_end   = d + train_cells + guard_cells + 1
            
            # 提取滑窗
            window_data = matrix[r_start:r_end, d_start:d_end]
            
            # 去掉保护区
            guard_r_start = r - guard_cells
            guard_r_end   = r + guard_cells + 1
            guard_d_start = d - guard_cells
            guard_d_end   = d + guard_cells + 1
            # 将保护区内的值置为None或移除
            # 方案1：先flatten全部，再排除保护区对应index
            window_flat = window_data.flatten()
            
            # 计算保护区相对索引并移除
            guard_zone = matrix[guard_r_start:guard_r_end, guard_d_start:guard_d_end].flatten()
            
            # 构造一个去除保护区的list
            # (这只是简化做法, 也可更优雅地使用mask等)
            train_list = []
            idx = 0
            for val in window_flat:
                if val not in guard_zone:
                    train_list.append(val)
                else:
                    # 为了避免'val not in guard_zone'的重复匹配问题,
                    # 这里更好做法是先把保护区index找出来再排除,
                    # 本例仅演示思路, 不考虑重复数值带来的影响.
                    pass
            train_list = np.array(train_list)
            noise_est = np.mean(train_list) if len(train_list)>0 else 0
            
            threshold = alpha * noise_est
            if matrix[r, d] > threshold:
                cfar_mask[r, d] = 1
    
    return cfar_mask

if __name__ == "__main__":
    # ====== 测试1D CA-CFAR ======
    np.random.seed(0)
    N = 200
    noise = np.random.exponential(scale=1, size=N)  # 指数分布噪声（常见非相干检测）
    signal_1d = noise.copy()
    # 人为加几个目标峰值
    targets_pos = [40, 90, 150]
    for pos in targets_pos:
        signal_1d[pos] += 10  # 增加较大的回波

        
    # 应用1D CA-CFAR
    cfar_mask_1d = ca_cfar_1d(signal_1d, guard_cells=2, train_cells=5, alpha=4)
    detected_indices_1d = np.where(cfar_mask_1d==1)[0]

    # 可视化1D结果
    plt.figure(figsize=(10,4))
    plt.plot(signal_1d, label='Signal + Noise')
    plt.plot(detected_indices_1d, signal_1d[detected_indices_1d], 'r^', label='CFAR Detections')
    plt.title('1D CA-CFAR Demo')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.legend()


    # ====== 测试2D CA-CFAR ======
    # 模拟一个距离 x 多普勒的噪声背景
    nr, nd = 64, 64
    noise_2d = np.random.exponential(scale=1, size=(nr, nd))
    signal_2d = noise_2d.copy()

    # 人为加几个目标点
    targets_2d = [(10, 10), (20, 45), (50, 30)]
    for (rr, dd) in targets_2d:
        signal_2d[rr, dd] += 15  # 增加明显的回波
        
    # 应用2D CA-CFAR
    cfar_mask_2d = ca_cfar_2d(signal_2d, guard_cells=1, train_cells=2, alpha=4)
    det_r, det_d = np.where(cfar_mask_2d==1)

    # 可视化2D结果
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title('Signal + Noise (Log scale)')
    plt.imshow(10*np.log10(signal_2d+1e-6), cmap='jet')
    plt.colorbar(label='dB')
    plt.subplot(1,2,2)
    plt.title('CFAR Detection Result')
    plt.imshow(cfar_mask_2d, cmap='gray')
    plt.colorbar(label='Detection=1')
    plt.show()