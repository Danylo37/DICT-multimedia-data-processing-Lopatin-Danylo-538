import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

model = keras.models.load_model('model.keras')
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

output_cars = []
output_planes = []
index_car = None
index_plane = None

for i in range(1, 51):
    image_car = keras.preprocessing.image.load_img('./Validation/cars/' + str(i) + '.jpg', target_size=(224, 224))
    image_plane = keras.preprocessing.image.load_img('./Validation/planes/' + str(i) + '.jpg', target_size=(224, 224))

    img = np.array(image_car)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)
    label = model.predict(img)
    output_cars.append(100 if float(f'{label[0][0]:.0f}') == 0 else 0)
    if index_car is None and float(f'{label[0][0]:.0f}') == 0:
        index_car = i - 1

    img = np.array(image_plane)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)
    label = model.predict(img)
    output_planes.append(100 if float(f'{label[0][0]:.0f}') == 1 else 0)
    if index_plane is None and float(f'{label[0][0]:.0f}') == 1:
        index_plane = i - 1

print(f"Сережня ймовірність вірного визначення автомобіля на зображенні = {np.mean(output_cars):.2f} %")
print(f"Сережня ймовірність вірного визначення літака на зображенні = {np.mean(output_planes):.2f} %")

image_plane_for_test = keras.preprocessing.image.load_img('./Validation/planes/' + str(index_plane + 1) + '.jpg',
                                                          target_size=(224, 224))
img = np.array(image_plane_for_test)
img = img / 255.0
img = img.reshape(1, 224, 224, 3)

result = []
sigma_x = []
result_s = []

for sigma in np.arange(0, 1, 0.1):
    for i in range(0, 30):
        image_plane_for_test_noisy = img + np.random.normal(loc=0.0, scale=sigma ** 0.5, size=img.shape)
        image_plane_for_test_noisy = np.clip(image_plane_for_test_noisy, 0., 1.)
        label = model.predict(image_plane_for_test_noisy)
        result_s.append(float(f'{label[0][0]:.2f}'))
    result.append(np.mean(result_s) * 100)
    sigma_x.append(sigma ** 0.5)

plt.figure(figsize=(12, 8))
plt.plot(sigma_x, result)
plt.xlabel('σ', fontsize=14)
plt.ylabel("Ймовірність вірного визначення класу об'єкту", fontsize=14)
plt.grid(which='both', linestyle='--', linewidth=0.2, color='gray')
plt.show()
