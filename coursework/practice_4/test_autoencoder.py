import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np


def main():
    model = keras.models.load_model('filter_model.keras')

    variant = 4
    sigma = np.sqrt(variant/100)

    (_, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    np.random.seed(42)
    x_test_noisy = x_test + np.random.normal(loc=0.0, scale=sigma, size=x_test.shape)
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

    reconstructed_images = model.predict(x_test_noisy)

    indices = [variant, variant + 500, variant + 1000, variant + 1500]
    titles = ["Original image", "Corrupted image", "Predicted image"]

    plt.figure(figsize=(9, 12))

    for i, idx in enumerate(indices):
        plt.subplot(len(indices), 3, i * 3 + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        if i == 0:
            plt.title(titles[0])

        plt.subplot(len(indices), 3, i * 3 + 2)
        plt.imshow(x_test_noisy[idx].reshape(28, 28), cmap='gray')
        if i == 0:
            plt.title(titles[1])

        plt.subplot(len(indices), 3, i * 3 + 3)
        plt.imshow(reconstructed_images[idx].reshape(28, 28), cmap='gray')
        if i == 0:
            plt.title(titles[2])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()