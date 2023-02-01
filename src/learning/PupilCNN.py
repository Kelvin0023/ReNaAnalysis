# load in the images
import os
import pickle

import numpy as np
from PIL import Image
from keras import layers
from rena.utils.data_utils import RNStream
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras

image_dir = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/3-5-2022/0/ReNaUnityCameraCapture_03-05-2022-13-42-32'
data_file = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/3-5-2022/0/03_05_2022_13_43_31-Exp_ReNaEEG-Sbj_SK-Ssn_0.dats'

print("Loading pupil")
data = RNStream(data_file).stream_in(ignore_stream=('monitor1'), jitter_removal=False)
y = data['Unity.VarjoEyeTrackingComplete'][0][24, :]

image_list = []
for i, fn in enumerate(os.listdir(image_dir)):
    print('Loading image {0} of {1}'.format(i + 1, len(os.listdir(image_dir))))
    image = np.asarray(Image.open(os.path.join(image_dir, fn)))
    image_list.append(image)
    break
x = np.array(image_list)
# TODO remove alpha channel if there is one
if x.shape[-1] == 4:
    x = x[:, :, :, :3]
x = x / 255  # min max norm

# match the timestamps
timestamps = data['Unity.VarjoEyeTrackingComplete'][1]
timestamp_end_idx = int(len(timestamps) * (len(x) / len(os.listdir(image_dir))))
x_timestamps = np.linspace(timestamps[0], timestamps[timestamp_end_idx], len(x))

# create the input sequence
# every 20 frames, use the previous 20 frames to infer the pupil size
step_size = 20
window_size = 20
X = []
Y = []
for i in range(window_size, len(x), step_size):
    X.append(x[i-window_size:i])
    # find the corresponding y
    Y.append(y[np.argmin(abs(timestamps - x_timestamps[i+1]))])
X = np.array(X)
Y = np.array(Y)
Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))  # min-max normalize Y

# if X and Y are preloaded
X = pickle.dump(X, open('X.npy', 'wb'))
Y = pickle.dump(Y, open('Y.npy', 'wb'))
X = pickle.load(open('X.npy', 'rb'))
Y = pickle.load(open('Y.npy', 'rb'))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

input = layers.Input(shape=X_train.shape[1:])
x = layers.ConvLSTM2D(
    filters=8,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(input)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x)

# x = layers.ConvLSTM2D(
#     filters=8,
#     kernel_size=(3, 3),
#     padding="same",
#     return_sequences=True,
#     activation="relu",
# )(x)
# x = layers.BatchNormalization()(x)
# x = layers.ConvLSTM2D(
#     filters=64,
#     kernel_size=(1, 1),
#     padding="same",
#     return_sequences=True,
#     activation="relu",
# )(x)
x = layers.Flatten()(x)

x = layers.Dense(units=256, activation='relu')(x)
x = layers.Dense(units=128, activation='relu')(x)
x = layers.Dense(units=64, activation='relu')(x)
x = layers.Dense(units=32, activation='relu')(x)
x = layers.Dense(units=1)(x)

model = keras.models.Model(input, x)
model.compile(
    loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
)

# Define modifiable training hyperparameters.
epochs = 100
# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
batch_size = 2

# Fit the model to the training data.
model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, Y_test),
    callbacks=[early_stopping, reduce_lr],
)