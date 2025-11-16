import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import tensorflow as tf
import math

img_width, img_height = 224, 224
train_data_dir = './Train'
validation_data_dir = './Validation'
nb_train_samples = 400
nb_validation_samples = 100
epochs = 30
batch_size = 16

if keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

input_image = keras.layers.Input(shape=input_shape)
x = keras.layers.Conv2D(32, (2, 2), input_shape=input_shape)(input_image)
x = keras.layers.Activation('relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Conv2D(32, (2, 2), input_shape=input_shape)(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Conv2D(32, (2, 2))(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Conv2D(64, (2, 2))(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64)(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(32)(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(1)(x)
x = keras.layers.Activation('sigmoid')(x)

model = keras.models.Model(input_image, x)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / batch_size),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=math.ceil(validation_generator.samples / batch_size))

model.save('model.keras')
