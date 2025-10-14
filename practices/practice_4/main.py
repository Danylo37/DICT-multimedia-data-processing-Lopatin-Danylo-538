import numpy as np
import matplotlib.pyplot as plt


def main():
    dt = 0.001
    t = np.arange(0, 1, dt)
    n = t.size
    f1 = 50.0
    f2 = 120.0

    x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

    np.random.seed(42)
    std = 2.5
    noise = np.random.normal(0, std, n)
    x_n = x + noise

    y = np.fft.fft(x_n)

    PSD = y * np.conj(y) / n

    threshold = 100
    y_filtered = y.copy()
    y_filtered[PSD < threshold] = 0

    f_inp = np.fft.ifft(y_filtered).real

    plt.plot(t, x_n,  color='c', linewidth=1.5, label = 'Noisy signal')
    plt.plot(t, x,  color='k', linewidth=2.0, label = 'Input signal')
    plt.xlim(t[0], t[-1])
    plt.legend()
    plt.show()

    plt.plot(t, f_inp,  color='k', linewidth=2.0, label = 'Input signal')
    plt.xlim(t[0], t[-1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()