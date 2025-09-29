# Моделювання небілого шумоподібного сигналу та аналіз кореляційної функції
# Завдання
# 1. Згенерувати адитивний білий гауссів шум (1024 значення, мат. очікування дорівнює 1, sigma = 1).
# 2. Застосувати до генерованої послідовності лінійний фільтр з розмірами вікон w = 3, 5, 7.
# 3. Отримати автокореляційну функцію (АКФ).
# 4. Виконати зсув послідовності на 5 відліків і побудувати крос-кореляційну функцію для w = 3, 5, 7.

import numpy as np
import matplotlib.pyplot as plt


def generate_noise(N, mean, sigma, seed):
    np.random.seed(seed)
    return np.random.normal(loc=mean, scale=sigma, size=N)


def moving_average(x, w):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')


def unbiased_acf(x):
    N = len(x)

    x = np.asarray(x, dtype=np.float64)

    x_mean = np.mean(x)
    x_centered = x - x_mean

    corr = np.correlate(x_centered, x_centered, mode='full')
    lags = np.arange(-N + 1, N)

    center_idx = N - 1
    r0 = corr[center_idx]

    if r0 == 0:
        return lags, corr

    acf_normalized = corr / r0
    return lags, acf_normalized


def cross_correlation(x, y):
    N = len(x)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    xc = x - x_mean
    yc = y - y_mean

    corr = np.correlate(xc, yc, mode='full')
    lags = np.arange(-N + 1, N)

    rxx0 = np.sum(xc * xc) / N
    ryy0 = np.sum(yc * yc) / N

    denom_norm = np.sqrt(rxx0 * ryy0)
    if denom_norm == 0:
        return lags, corr

    c_normalized = corr / denom_norm
    return lags, c_normalized


def plot_time_series(x, title):
    plt.figure()
    plt.plot(x, linewidth=0.8)
    plt.title(title)
    plt.xlabel('Відлік')
    plt.ylabel('Амплітуда')
    plt.grid(True)


def plot_correlation(lags, corr, title):
    plt.figure()
    plt.plot(lags, corr, linewidth=0.9)
    plt.axvline(0, color='k', linewidth=0.6, linestyle='--')
    plt.title(title)
    plt.xlabel('Відліки')
    plt.ylabel('Нормалізована кореляція')
    plt.grid(True)


def main():
    N = 1024
    mean = 1.0
    sigma = 1.0
    seed = 0
    windows = [3, 5, 7]
    shift_k = 5

    noise = generate_noise(N=N, mean=mean, sigma=sigma, seed=seed)
    plot_time_series(noise, f'Адитивний білий гауссів шум (N={N}, mean={mean}, sigma={sigma})')

    filtered_signals = {}
    for w in windows:
        filtered = moving_average(noise, w)
        filtered_signals[w] = filtered
        plot_time_series(filtered, f'Відфільтрований сигнал, w={w}')

    for w in windows:
        sig = filtered_signals[w]
        lags, acf = unbiased_acf(sig)
        plot_correlation(lags, acf, f'Автокореляційна функція, w={w}')

    for w in windows:
        sig = filtered_signals[w]
        sig_shifted = np.roll(sig, shift_k)
        lags, ccf = cross_correlation(sig, sig_shifted)
        plot_correlation(lags, ccf, f'Крос-кореляційна функція (w={w}, зсув={shift_k})')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
