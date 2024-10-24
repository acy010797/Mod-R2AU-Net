import pandas as pd
import numpy as np
import matplotlib as mtp
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import tensorflow as tf

from glob import glob

msk_paths = glob('https://drive.google.com/drive/folders/mri')
img_paths = [m.replace('_mask','') for m in msk_paths]

import cv2

rows, cols = 20, 2

fig = plt.figure(figsize = (4, 40))
for i in range(1, rows*cols, 2):
    
    fig.add_subplot(rows, cols, i)
    img = cv2.imread(img_paths[i])
    plt.imshow(img)

    fig.add_subplot(rows, cols, i+1)
    msk = cv2.imread(msk_paths[i])
    plt.imshow(msk)
plt.savefig('my_image.png')
from sklearn.model_selection import train_test_split

mri_df_train, mri_df_test = train_test_split(mri_df, test_size = 0.1,random_state=77_47)
mri_df_train, mri_df_val = train_test_split(mri_df_train, test_size = 0.1,random_state=77_47)

mri_df_train = mri_df_train.reset_index(drop=True)
mri_df_val = mri_df_val.reset_index(drop=True)
mri_df_test = mri_df_test.reset_index(drop=True)

from keras.preprocessing.image import ImageDataGenerator

# assuming an human precision of 1% for manual tasks, so I'll use 2% for a safaty error margin
BATCH_SIZE = 16

def transformator(df,
                  rotation_range=0.2,
                  width_shift_range=0.02,
                  height_shift_range=0.02,
                  shear_range=0.1,
                  zoom_range=0.1,
                  horizontal_flip=True,
                  IMG_SIZE_TRAIN = (128,128),
                  BATCH_SIZE = 16):

  img_datag = ImageDataGenerator(
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                shear_range=shear_range,
                zoom_range=zoom_range,
                horizontal_flip=True,
                fill_mode='nearest')

  msk_datag = ImageDataGenerator(
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                shear_range=shear_range,
                zoom_range=zoom_range,
                horizontal_flip=True,
                fill_mode='nearest')

  img_generated = img_datag.flow_from_dataframe(
                      df,
                      x_col = 'img',
                      class_mode = None,
                      color_mode = 'rgb',
                      target_size = IMG_SIZE_TRAIN,
                      batch_size = BATCH_SIZE,
                      save_to_dir = None,
                      save_prefix = 'image',
                      seed = 777447)

  msk_generated = msk_datag.flow_from_dataframe(
                      df,
                      x_col='msk',
                      class_mode=None,
                      color_mode='grayscale',
                      target_size=IMG_SIZE_TRAIN,
                      batch_size=BATCH_SIZE,
                      save_to_dir=None,
                      save_prefix='mask',
                      seed=777447)

  generated = zip(img_generated, msk_generated)

  for img, msk in generated:
      img = img / 255
      msk = msk / 255

      msk[msk > 0.5] = 1
      msk[msk <= 0.5] = 0

      yield (img,msk)

data_train = transformator(mri_df_train)
data_val = transformator(mri_df_val)
data_test = transformator(mri_df_test)


# R2CL Block (Recurrent Residual Convolutional Layer)
def R2CL_block(inputs, filters, t=2):
    x = inputs
    for _ in range(t):
        residual = x
        x = layers.Conv2D(filters, (3, 3), padding="same", activation='relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding="same", activation='relu')(x)
        x = layers.Add()([residual, x])
    return x

# Attention Gate
def attention_gate(x, g, filters):
    x = layers.Conv2D(filters, (1, 1), padding="same")(x)
    g = layers.Conv2D(filters, (1, 1), padding="same")(g)
    add = layers.Add()([x, g])
    act = layers.Activation('relu')(add)
    psi = layers.Conv2D(1, (1, 1), padding="same")(act)
    psi = layers.Activation('sigmoid')(psi)
    out = layers.Multiply()([x, psi])
    return out

# Downsampling path
def downsample_block(inputs, filters):
    r2cl = R2CL_block(inputs, filters)
    pool = layers.MaxPooling2D((2, 2))(r2cl)
    return r2cl, pool

# Upsampling path
def upsample_block(inputs, skip_features, filters):
    up = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
    att = attention_gate(skip_features, up, filters)
    concat = layers.Concatenate()([up, att])
    r2cl = R2CL_block(concat, filters)
    return r2cl

# Mod-R2AU-Net model
def ModR2AU_Net(input_shape):
    inputs = layers.Input(input_shape)

    # Preprocessing: Gaussian filter, CLAHE, Resize (Simulated with Conv layer)
    x = layers.Conv2D(3, (3, 3), padding="same", activation='relu')(inputs)

    # Encoder (Downsampling path)
    skip1, pool1 = downsample_block(x, 16)
    skip2, pool2 = downsample_block(pool1, 32)
    skip3, pool3 = downsample_block(pool2, 64)
    skip4, pool4 = downsample_block(pool3, 128)

    # Bottleneck
    bottleneck = R2CL_block(pool4, 256)

    # Decoder (Upsampling path)
    up4 = upsample_block(bottleneck, skip4, 128)
    up3 = upsample_block(up4, skip3, 64)
    up2 = upsample_block(up3, skip2, 32)
    up1 = upsample_block(up2, skip1, 16)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs, outputs, name="Mod-R2AU-Net")
    return model

# Define input shape and create model
input_shape = (128, 128, 3)
model = ModR2AU_Net(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='IoU Loss', metrics=['accuracy', 'DiceCoefficient'])

# Model summary
model.summary()

import keras.backend as K

smooth=100

def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersec = K.sum(y_true * y_pred)
    mod_sum = K.sum(y_true) + K.sum(y_pred)

    return (2 * intersec + smooth) / (mod_sum + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersec = K.sum(y_true * y_pred)
    comb_area = K.sum(y_true + y_pred) - intersec

    return (intersec + smooth) / (comb_area + smooth)

def iou_loss(y_true, y_pred):
  return - iou(y_true, y_pred)

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

model.compile(optimizer = Adam(learning_rate=0.0001),
              loss = iou_loss,
              metrics=["binary_accuracy", iou, dice_coef])


history = model.fit(data_train,
                    steps_per_epoch=(len(mri_df_train) / BATCH_SIZE),
                    epochs=100,
                    callbacks=[ModelCheckpoint('unet_128_mri_seg.hdf5', verbose=1, save_best_only=True)],
                    validation_data = data_val,
                    validation_steps=len(mri_df_val) / BATCH_SIZE)

results = model.evaluate(data_test, steps=len(mri_df_test) / BATCH_SIZE)

print("Test loss: ",round(results[0],3),
      "Test Binary Accuracy: ",round(results[1],3),
      "Test IOU: ",round(results[2],3),
      "Test Dice Coefficent: ",round(results[3],3))

for i in range(40):
    index=np.random.randint(1,len(mri_df_test.index))
    img = cv2.imread(mri_df_test['img'].iloc[index])
    img = cv2.resize(img ,(128, 128))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred=model.predict(img)

    plt.figure(figsize=(12,12))
    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(img))
    plt.title('True Image')
    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(cv2.imread(mri_df_test['msk'].iloc[index])))
    plt.title('True Mask')
    plt.subplot(1,3,3)
    plt.imshow(np.squeeze(pred) > .5)
    plt.title('Prediction')
    plt.show()
plt.savefig('my_image_out.png')

def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union (IoU) metric.
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        IoU score.
    """
    y_true = tf.cast(y_true, tf.float32)  # Cast ground truth mask to float32
    y_pred = tf.cast(y_pred, tf.float32)  # Cast predicted mask to float32

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])  # Element-wise multiplication for intersection
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection  # Union calculation

    iou = tf.reduce_mean((intersection + smooth) / (union + smooth), axis=0)  # Mean IoU over the batch
    return iou

# Compile model with additional IoU metric
model.compile(optimizer='adam', 
              loss='IoU Loss', 
              metrics=['accuracy', dice_coefficient, iou_metric])

# Training the model (example with placeholders for dataset)
# Replace 'train_dataset' and 'val_dataset' with your actual data
history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    epochs=50,  # Example epochs
                    batch_size=16)

# Function to plot the accuracy, loss, Dice Coefficient, and IoU during training
def plot_metrics(history):
    """
    Plots training and validation accuracy, loss, Dice coefficient, and IoU over epochs.
    Args:
        history: Training history object from model.fit.
    """
    # Extract values from history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    dice = history.history['dice_coefficient']
    val_dice = history.history['val_dice_coefficient']
    iou = history.history['iou_metric']
    val_iou = history.history['val_iou_metric']

    epochs = range(1, len(acc) + 1)

    # Plot Accuracy
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Dice Coefficient
    plt.subplot(2, 2, 3)
    plt.plot(epochs, dice, 'b', label='Training Dice Coefficient')
    plt.plot(epochs, val_dice, 'r', label='Validation Dice Coefficient')
    plt.title('Training and Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    # Plot IoU
    plt.subplot(2, 2, 4)
    plt.plot(epochs, iou, 'b', label='Training IoU')
    plt.plot(epochs, val_iou, 'r', label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    # Display all plots
    plt.tight_layout()
    plt.show()

plot_metrics(history)
