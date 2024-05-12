# EX 7Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
The image dataset we taken is mnist and the model must denoise the images and show it with better quality and remove the unwanted noises and learn to show the better version of the images.

Autoencoder is an unsupervised artificial neural network that is trained to copy its input to output. An autoencoder will first encode the image into a lower-dimensional representation, then decodes the representation back to the image.The goal of an autoencoder is to get an output that is identical to the input. MNIST is a dataset of black and white handwritten images of size 28x28.Denoising is the process of removing noise. This can be an image, audio, or document.These noisy digits will serve as our input data to our encoder. Autoencoders uses MaxPooling, convolutional and upsampling layers to denoise the image.

![image](https://github.com/vidhyasrikachapalayam/convolutional-denoising-autoencoder/assets/119477817/d8252924-eae5-4a2d-b82b-80e077365892)


## DESIGN STEPS
Step 1:
Import the necessary libraries and dataset.

Step 2:
Load the dataset and scale the values for easier computation.

Step 3:
Add noise to the images randomly for both the train and test sets.

Step 4:
Build the Neural Model using

Convolutional Layer
Pooling Layer
Up Sampling Layer.
Make sure the input shape and output shape of the model are identical.
Step 5:
Pass test data for validating manually.

Step 6:
Plot the predictions for visualization

## PROGRAM
```
Developed by: VIDHYASRI.K
Register no: 212222230170

Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers, utils, models
from tensorflow.keras.datasets import mnist

Read the dataset and scale it
(x_train, _), (x_test, _) = mnist.load_data()

x_train.shape

x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.

x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

Add noise to image
noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor*np.random.normal(loc=0.0, scale=1.0,
                                                               size=x_train_scaled.shape)
x_test_noisy = x_test_scaled + noise_factor*np.random.normal(loc=0.0, scale=1.0,
                                                             size=x_test_scaled.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
Plot the images
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

Develop an Autoencoder DL Model
input_img = keras.Input(shape=(28, 28, 1))

x=layers.Conv2D(16,(5,5),activation='relu',padding='same')(input_img)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(7,7),activation='relu',padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(encoded)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(5,5),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(5,5),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(5,5),activation='relu')(x)
x=layers.UpSampling2D((1,1))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)

autoencoder.summary()

Compile and Fit the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=3,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))

Plot Metrics Graph
metrics = pd.DataFrame(autoencoder.history.history)
metrics[['loss','val_loss']].plot()

Predict Using the model
decoded_imgs = autoencoder.predict(x_test_noisy)

Plot the original, noisy & reconstructed images
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## OUTPUT
## Training Loss, Validation Loss Vs Iteration Plot:

![image](https://github.com/vidhyasrikachapalayam/convolutional-denoising-autoencoder/assets/119477817/0ccbe9d9-2cc9-42ea-9753-c5033fb5981b)

### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/vidhyasrikachapalayam/convolutional-denoising-autoencoder/assets/119477817/cb398c11-4b55-45e2-b7fc-c9d3eefe8dd8)

## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.
