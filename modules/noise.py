import numpy as np
def awgn(x, snr, out='signal', method='vectorized', axis=0):
    '''
    添加噪声,由matlab更改而来
    https://www.cnblogs.com/minyuan/p/14078114.html
    '''
    # Signal power
    if method == 'vectorized':
        N = x.size
        Ps = np.sum(x ** 2 / N)

    elif method == 'max_en':
        N = x.shape[axis]
        Ps = np.max(np.sum(x ** 2 / N, axis=axis))

    elif method == 'axial':
        N = x.shape[axis]
        Ps = np.sum(x ** 2 / N, axis=axis)
    else:
        raise ValueError('method \"' + str(method) + '\" not recognized.')
    # Signal power, in dB
    Psdb = 10 * np.log10(Ps)
    # Noise level necessary
    Pn = Psdb - snr
    # Noise vector (or matrix)
    n = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, x.shape)
    if out == 'signal':
        return x + n
    elif out == 'noise':
        return n
    elif out == 'both':
        return x + n, n
    else:
        return x + n
    

def aexpn(signal, snr):
    """
    给信号添加指数噪声。
    
    参数:
    - signal: 输入的信号（例如 Rx）
    - SNR_dB: 目标信噪比（以分贝为单位）
    
    返回:
    - 含噪声的信号
    """
    SNR_linear = 10 ** (snr/ 10)
    # 计算信号功率
    Ps = np.mean(signal**2)
    N = len(signal)
    
    # 生成指数噪声
    noise_exp = np.random.exponential(scale=1, size=N)
    noise_power = np.mean(noise_exp**2)
    
    # 计算噪声尺度因子
    scale_factor = np.sqrt(Ps / (SNR_linear * noise_power))
    
    # 返回添加噪声后的信号
    return signal + scale_factor * noise_exp


if __name__ == "__main__":
    print('test')
