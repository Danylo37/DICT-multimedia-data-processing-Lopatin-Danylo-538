import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title, L=255):
    plt.figure(figsize=(5, 5))
    plt.imshow(image.astype(np.uint8), cmap="gray", vmin=0, vmax=L)
    plt.title(title)
    plt.show()


def plot_hist_with_cumulative(title, bins, count, cs):
    plt.subplot(2, 1, 1)
    plt.bar(bins[:-1], count)
    plt.title(title)
    plt.subplot(2, 1, 2)
    plt.plot(cs / cs.max())
    plt.tight_layout()
    plt.show()


def compute_stats(image, L=255):
    Imin = float(np.amin(image))
    Imax = float(np.amax(image))
    K = (Imax - Imin) / L
    return Imin, Imax, K


def compute_histogram(image, M=10, range_vals=(0, 256)):
    count, bins = np.histogram(image.flatten(), M, range_vals)
    cs = count.cumsum()
    cs_norm = cs * float(max(count)) / max(cs)
    return count, bins, cs_norm


def main():
    # --------------------------- Practice 1 ---------------------------
    image = plt.imread("image.jpg")

    weights = [0.299, 0.587, 0.114]
    I = np.dot(image, weights)
    L = 255

    Imin, Imax, K = compute_stats(I)
    show_image(I, f"Original, K = {K:.2f}")

    low_out = 25
    high_out = 190
    gamma = 1.3

    Im_s = low_out + (high_out - low_out) * ((I - Imin) / (Imax - Imin)) ** gamma

    Imin_s, Imax_s, K_s = compute_stats(Im_s)
    show_image(Im_s, f"Result1, Imax = {Imax_s:.2f}, Imin = {Imin_s:.2f}, K = {K_s:.2f}")

    Im_n = Imax_s - Im_s
    Imin_n, Imax_n, K_n = compute_stats(Im_n)
    show_image(Im_n, f"Negative, Imax = {Imax_n:.2f}, Imin = {Imin_n:.2f}, K = {K_n:.2f}")

    # --------------------------- Practice 2 ---------------------------
    count, bins, cs_norm = compute_histogram(I)

    plot_hist_with_cumulative("Histogram and Cumulative Function", bins, count, cs_norm)

    distension = np.round((L / (Imax - Imin)) * (I - Imin)).astype(I.dtype)
    distension[I < Imin] = 0
    distension[I > Imax] = L

    Imin_dist, Imax_dist, K_dist = compute_stats(distension)
    show_image(distension, f"Distension, Imax = {Imax_dist:.2f}, Imin = {Imin_dist:.2f}, K = {K_dist:.2f}")

    count_dist, bins_dist, cs_norm_dist = compute_histogram(distension)

    plot_hist_with_cumulative("Distension Histogram and Cumulative Function", bins_dist, count_dist, cs_norm_dist)

    count_full, bins_full = np.histogram(I.flatten(), 256, (0, 256))
    cs_full = count_full.cumsum()
    cs_m = np.ma.masked_equal(cs_full, 0)
    cs_m = (cs_m - cs_m.min()) * L / (cs_m.max() - cs_m.min())
    cs_final = np.ma.filled(cs_m, 0).astype('uint8')
    I_eq = cs_final[I.astype(np.uint8)]

    Imin_eq, Imax_eq, K_eq = compute_stats(I_eq)
    show_image(I_eq, f"Equalization, Imax = {Imax_eq:.2f}, Imin = {Imin_eq:.2f}, K = {K_eq:.2f}")

    count_eq, bins_eq, cs_norm_eq = compute_histogram(I_eq)

    plot_hist_with_cumulative("Equalization Histogram and Cumulative Function", bins_eq, count_eq, cs_norm_eq)


if __name__ == "__main__":
    main()
