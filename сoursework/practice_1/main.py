import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title, L):
    plt.figure(figsize=(5, 5))
    plt.imshow(image.astype(np.uint8), cmap="gray", vmin=0, vmax=L)
    plt.title(title)
    plt.show()


def compute_stats(image, L):
    Imin = float(np.amin(image))
    Imax = float(np.amax(image))
    K = (Imax - Imin) / L
    return Imin, Imax, K


def main():
    # Task 1
    image = plt.imread("image.jpg")

    # Task 2
    weights = [0.299, 0.587, 0.114]
    I = np.dot(image, weights)
    L = 255

    Imin, Imax, K = compute_stats(I, L)
    show_image(I, f"Original, K = {K:.2f}", L)

    # Task 3
    low_out = 25
    high_out = 190
    gamma = 1.3

    Im_s = low_out + (high_out - low_out) * ((I - Imin) / (Imax - Imin)) ** gamma

    Imin_s, Imax_s, K_s = compute_stats(Im_s, L)
    show_image(Im_s, f"Result1, Imax = {Imax_s:.2f}, Imin = {Imin_s:.2f}, K = {K_s:.2f}", L)

    # Task 4
    Im_n = Imax_s - Im_s
    Imin_n, Imax_n, K_n = compute_stats(Im_n, L)
    show_image(Im_n, f"Negative, Imax = {Imax_n:.2f}, Imin = {Imin_n:.2f}, K = {K_n:.2f}", L)


if __name__ == "__main__":
    main()
