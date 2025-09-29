import numpy as np
import matplotlib.pyplot as plt


def normalize(data):
    return (data - np.mean(data)) / np.std(data)


def display_plot(signal_data, plot_title):
    plt.plot(signal_data)
    plt.title(plot_title)
    plt.show()


size = 1024
w = 5
snr = 10

x = np.random.randn(size)

filter_kernel = np.ones(w) / w
filtered_signal = np.convolve(x, filter_kernel, mode='same')
display_plot(filtered_signal, 'Сигнал після фільтрації')

x_norm = normalize(filtered_signal)
display_plot(x_norm, 'Сигнал після нормалізації')

y_n1 = normalize(np.random.randn(size))
display_plot(y_n1, 'Нормалізований шум y_n1')

y_n2 = normalize(np.random.randn(size))
display_plot(y_n2, 'Нормалізований шум y_n2')

signal_variance = np.var(x_norm)
noise_variance = signal_variance / (10 ** (snr / 10))
required_noise_std = np.sqrt(noise_variance)

y_n1_scaled = y_n1 * required_noise_std
snr_1 = 10 * np.log10(np.var(x_norm) / np.var(y_n1_scaled))
print(f"SNR для сигналу y_n1: {snr_1}dB")

y_n2_scaled = y_n2 * required_noise_std
snr_2 = 10 * np.log10(np.var(x_norm) / np.var(y_n2_scaled))
print(f"SNR для сигналу y_n2: {snr_2}dB")

x_n1 = x_norm + y_n1_scaled
display_plot(x_n1, 'Суміш сигналу x та шуму n1')

x_n2 = x_norm + y_n2_scaled
display_plot(x_n2, 'Суміш сигналу x та шуму n2')

l = len(x_n1)
correlation_full = np.correlate(x_n1, x_n2, mode='full')
lags = np.arange(-l + 1, l)
denominator = l - np.abs(lags)
correlation_function = correlation_full / denominator

plt.plot(lags, correlation_function)
plt.title('Взаємно кореляційна функція для сигналів x_n1 і x_n2')
plt.show()
