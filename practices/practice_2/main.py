import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, w):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')


def normalize(x):
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return (x - mu) / sigma


def required_noise_std(signal_var, snr_db):
    noise_var = signal_var / (10 ** (snr_db / 10))
    return np.sqrt(noise_var)


def cross_correlation_unbiased(x, y):
    length = len(x)
    full = np.correlate(x, y, mode='full')
    lags = np.arange(-length + 1, length)
    denom = (length - np.abs(lags)).astype(float)
    unbiased = full / denom
    return lags, unbiased


def plot_signal(data, title, x_data=None):
    plt.figure(figsize=(10, 6))
    if x_data is not None:
        plt.plot(x_data, data)
    else:
        plt.plot(data)
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    size = 1024
    w = 5
    snr_db = 10

    x_white = np.random.randn(size)

    plot_signal(x_white[:200], 'White noise signal')

    x_filtered = moving_average(x_white, w)

    plot_signal(x_filtered[:200], 'Signal after filtering')

    x = normalize(x_filtered)

    plot_signal(x[:200], 'Signal after normalization')

    y_n1 = np.random.randn(size)
    y_n1 = normalize(y_n1)

    y_n2 = np.random.randn(size)
    y_n2 = normalize(y_n2)

    plot_signal(y_n1[:200], 'Normalized noise signal y_n1')

    plot_signal(y_n2[:200], 'Normalized noise signal y_n2')

    sigma_sqr = np.var(x)
    sigma_noise_required = required_noise_std(sigma_sqr, snr_db)

    y_n1_scaled = y_n1 * sigma_noise_required
    y_n2_scaled = y_n2 * sigma_noise_required

    snr_1 = 10 * np.log10(np.float64(np.var(x)) / np.var(y_n1_scaled))
    snr_2 = 10 * np.log10(np.float64(np.var(x)) / np.var(y_n2_scaled))

    print(f"SNR for signal 1: {snr_1:.2f} dB")
    print(f"SNR for signal 2: {snr_2:.2f} dB")

    x_n1 = x + y_n1_scaled
    x_n2 = x + y_n2_scaled

    plot_signal(x_n1[:200], 'Signal mixture x + n1')

    plot_signal(x_n2[:200], 'Signal mixture x + n2')

    lags, cross_corr = cross_correlation_unbiased(x_n1, x_n2)

    plot_signal(cross_corr, 'Cross-correlation function', lags)


if __name__ == '__main__':
    main()
