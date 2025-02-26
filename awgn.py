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
    
if __name__ == "__main__":
    print('test')
