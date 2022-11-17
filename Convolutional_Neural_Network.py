import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import load_img, img_to_array

# rescale => untuk perskalaan fitur setiap pixel ( pixel mengambil nilai dari 0 sampai 255 )
# shear_range => Shearing image skala 0.2
# zoom_range => Zooming image dengan range 0.2

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# target_size => ukuran gambar (64,64) => 64 x 64

train_set = train_datagen.flow_from_directory(
    "assets/dataset/training_set",
    target_size=(64,64),
    batch_size=32,
    class_mode="binary"
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

# target_size => ukuran gambar (64,64) => 64 x 64

test_set = test_datagen.flow_from_directory(
    "assets/dataset/test_set",
    target_size=(64,64),
    batch_size=32,
    class_mode="binary"
)

# Make the model (Convolutional), Create Pooling and Flattening

CNN = tf.keras.models.Sequential([
    # Jika gambar hitam putih cukup dimensi 1 pada input_shape, jika berwarna dimensi 3
    # Fitur => untuk mendetaksi jumlah fitur yang ingin di terapkan pada gambar
    # kernel_size => kaya jumlah (3) 3 x 3 feature detector yang di gunakan untuk mengolah gambar
    # input_shape => menurut gua buat naro hasil data (feature maps) setelah featur detector di lakukan

    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation="relu",
        input_shape=[64, 64, 3]
    ),

    # Setelah feature map di buat maka akan di buat pooled feature map agar gambar gapat di baca dengan jelas oleh AI
    # pool_size => sama kaya kernel_zie 2 x 2 pooled feature detector
    # strides => untuk menggeser setiap frame sebanyak 2 pixel

    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    # karna sudah di lapisan kedua convolutional maka input_shape di hapus saja
    # karena secara otomatis akan mengikuti yang pertama

    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        activation="relu"
    ),

    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),

    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),

    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    # flattening adalah merubah dari matriks yang ada di pooling layer menjadi satu kolom saja (vector)
    # yang tadinya berupa matrix akan di ubah menjadi vector
    #                     [1, 0]                        [1]
    #                     [2, 0]                        [0]
    #                                                   [2]
    #                                                   [0]

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(254, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

EPOCS = 25

# validation_data => kaya langsung di evaluate kalo di cnn

CNN.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
CNN.fit(train_set, epochs=EPOCS, validation_data=test_set, verbose=1)

# Making a single prediction

PATH = "assets/dataset/single_prediction/cat_or_dog_3.jpg"

test_image = load_img(PATH, target_size=(64, 64))

print("\nTest Image : ", test_image)

test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = CNN.predict(test_image, batch_size=10)

train_set.class_indices

# 0 => CAT
# 1 => DOG

if result[0][0] == 1:
    prediction = "DOG"
else:
    prediction = "CAT"

print(prediction)