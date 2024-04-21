import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np

# Ensure TensorFlow is using GPU
print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0 = all messages are logged (default behavior)
                                          # 1 = INFO messages are not printed
                                          # 2 = INFO and WARNING messages are not printed
                                          # 3 = INFO, WARNING, and ERROR messages are not printed

tf.config.list_physical_devices('GPU')  # This line triggers GPU detection and logging
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Check and print the CUDA version TensorFlow is using
# print("CUDA version: ", tf.sysconfig.get_build_info()["cuda_version"])

# Check and print the cuDNN version TensorFlow is using
# print("cuDNN version: ", tf.sysconfig.get_build_info()["cudnn_version"])



# Encoder
encoder_input = Input(shape=(32, 32, 2), name='encoder_input')
x = layers.Conv2D(27, (3, 3), activation='relu', padding='same')(encoder_input)
x = layers.Conv2D(27, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(27, (3, 3), activation='relu', padding='same')(x)
x = layers.AveragePooling2D((2, 2))(x)
x = layers.Conv2D(27, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(27, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(27, (3, 3), activation='relu', padding='same')(x)
x = layers.AveragePooling2D((2, 2))(x)
x = layers.Conv2D(27, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(27, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(27, (3, 3), activation='relu', padding='same')(x)
x = layers.AveragePooling2D((4, 4))(x)
encoder_output = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)

encoder = models.Model(encoder_input, encoder_output, name="encoder")

# Decoder
decoder_input = Input(shape=(2, 2, 1), name='decoder_input')
x = layers.Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(decoder_input)
x = layers.UpSampling2D((4, 4))(x)
x = layers.Conv2DTranspose(27, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(27, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(27, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(27, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(27, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(27, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(27, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(27, (3, 3), activation='relu', padding='same')(x)
decoder_output = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = models.Model(decoder_input, decoder_output, name="decoder")

# Autoencoder
autoencoder_input = Input(shape=(32, 32, 2), name='autoencoder_input')
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = models.Model(autoencoder_input, decoded_img, name="autoencoder")

autoencoder.compile(optimizer='adam', loss='mse')

# Generating random training and testing data
x_train = np.random.rand(18000, 32, 32, 2)  # 18,000 samples for training
x_train1 = np.random.rand(18000, 32, 32, 2)  # 18,000 samples for training
x_test = np.random.rand(2000, 32, 32, 2)    # 2,000 samples for testing

print('data gen finished')
# Configurable training batch size
batch_size = 1024  # Feel free to adjust this value

# Train the autoencoder
autoencoder.fit(x_train, x_train1,
                epochs=10,
                batch_size=batch_size,
                validation_data=(x_test, x_test))

# The model is now training on randomly generated data with the specified batch size.
