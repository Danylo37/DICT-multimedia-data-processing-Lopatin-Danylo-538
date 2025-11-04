import numpy as np
import matplotlib.pyplot as plt


def create_dct_matrix(N):
    DCT = np.zeros((N, N))
    factor = np.sqrt(2 / N)

    for u in range(N):
        for x in range(N):
            if x == 0:
                DCT[u, x] = factor * (1 / np.sqrt(2))
            else:
                DCT[u, x] = factor * np.cos(np.pi * (u + 0.5) * x / N)

    return DCT


def dct_2d(f):
    DCT = create_dct_matrix(f.shape[0])
    return DCT @ f @ DCT.T


def idct_2d(f):
    DCT = create_dct_matrix(f.shape[0])
    return DCT.T @ f @ DCT


def main():
    np.random.seed(42)

    a = np.asarray(np.random.randn(8, 8))

    G = dct_2d(a)
    a_restored = idct_2d(G)

    images = {
        "Original Image": a,
        "DCT Coefficients": G,
        "Restored Image": a_restored
    }

    for title, img in images.items():
        plt.figure()
        plt.imshow(img)
        plt.title(title)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
