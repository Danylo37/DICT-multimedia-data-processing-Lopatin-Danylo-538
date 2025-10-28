from tensorflow import keras
import numpy as np


def main():
    variant = 4
    sigma = np.sqrt(variant/100)

    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    x_train_noisy = x_train + np.random.normal(loc=0.0, scale=sigma, size=x_train.shape)
    x_test_noisy = x_test + np.random.normal(loc=0.0, scale=sigma, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

    input_image = keras.layers.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = keras.models.Model(input_image, decoded)
    model.compile(optimizer='adadelta', loss='binary_crossentropy')

    model.fit(x_train_noisy, x_train, epochs=100, batch_size=128, shuffle=True, validation_data=(x_test_noisy, x_test))
    model.save('filter_model.keras')

    keras.utils.plot_model(model, "model.png", show_shapes=True)


if __name__ == '__main__':
    main()