import numpy as np


def create_dct_matrix(N):
    C = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            a_i = 1 / np.sqrt(2) if i == 0 else 1
            C[i, j] = np.sqrt(2 / N) * a_i * np.cos(i * (2 * j + 1) * np.pi / (2 * N))

    return C


def filter_dct(signal, window_size, threshold):
    filtered = np.zeros_like(signal)

    for start in range(0, len(signal), window_size):
        end = min(start + window_size, len(signal))
        w = signal[start:end]
        C = create_dct_matrix(len(w))

        y = C @ w
        y[np.abs(y) < threshold] = 0
        filtered[start:end] = C.T @ y

    return filtered


def main():
    N = 2048
    t = np.arange(0, 2, 2 / N)
    x = 1 * np.sin(2 * np.pi * (2 * t - 0))
    x += 0.5 * np.sin(2 * np.pi * (6 * t - 0.1))
    x += 0.1 * np.sin(2 * np.pi * (20 * t - 0.2))

    sigma = 5
    np.random.seed(42)
    noise = np.random.normal(0, sigma, N)
    x_noisy = x + noise

    for window_size in [8, 16]:
        for beta in [3, 4]:
            threshold = beta * sigma
            filtered = filter_dct(x_noisy, window_size, threshold)
            mse = np.mean((x - filtered) ** 2)
            print(f"Window {window_size}, β={beta}: MSE = {mse:.4f}")


if __name__ == "__main__":
    main()