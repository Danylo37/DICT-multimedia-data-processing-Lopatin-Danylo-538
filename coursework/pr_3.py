import random
import cv2
import numpy as np
from matplotlib import pyplot as plt


def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def apply_noise(image, mean, var, noise_type):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)

    if noise_type == 'additive':
        out = image + noise
    elif noise_type == 'multiplicative':
        out = image * noise
    else:
        raise ValueError("noise_type must be 'additive' or 'multiplicative'")

    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def apply_filter_and_display(noises, window_sizes, filter_func, filter_name):
    for noise_name, noise in noises.items():
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for idx, (label, size) in enumerate(window_sizes.items()):
            if filter_name == "Median":
                img_filtered = filter_func(noise, size)
            elif filter_name == "Average":
                img_filtered = filter_func(noise, (size, size))
            else:
                raise ValueError("filter_name must be 'Median' or 'Average'")

            axes[idx].imshow(img_filtered)
            axes[idx].set_title(f"Result {noise_name}, {filter_name} filter {size}x{size}")
            axes[idx].set_xlabel(label, fontsize=16)

        plt.tight_layout()
        plt.show()
        plt.close()


def main():
    img_original = plt.imread("image.jpg")

    variant = 4

    img_gauss = apply_noise(img_original, 0, variant / 100, 'additive')
    img_multi = apply_noise(img_original, 1, variant / 100, 'multiplicative')
    img_sp = sp_noise(img_original, 0.1 + abs(variant - 14) / 100)

    noises = {
        "Additive noise": img_gauss,
        "Multiplicative noise": img_multi,
        "Salt & Pepper noise": img_sp,
    }

    for title, img in ({"Original": img_original} | noises).items():
        plt.imshow(img)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.close()
    plt.show()

    window_sizes = {'а': 3, 'б': 5, 'в': 7, 'г': 11}

    apply_filter_and_display(noises, window_sizes, cv2.medianBlur, "Median")
    apply_filter_and_display(noises, window_sizes, cv2.blur, "Average")


if __name__ == '__main__':
    main()