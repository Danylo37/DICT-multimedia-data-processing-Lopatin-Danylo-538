import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct


def plot_signal(signal, title):
    plt.plot(signal)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    N = 128
    mean = 0
    std_dev = 1
    awgn = np.random.normal(mean, std_dev, N)

    awgn_dct = dct(awgn, type=2, norm='ortho')
    awgn_idct = idct(awgn_dct, type=2, norm='ortho')

    plot_signal(awgn, 'AWGN')
    plot_signal(awgn_dct, 'DCT')
    plot_signal(awgn_idct, 'IDCT')


if __name__ == '__main__':
    main()
