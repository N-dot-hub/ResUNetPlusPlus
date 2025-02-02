"""
Runs the process for ResUNet++, ResUNet or UNet.
Model architecture is imported from resunet_pp.py, resunet.py or unet.py as selected
"""

import os
import time
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.python.keras.metrics import (Precision, Recall, MeanIoU)
from tensorflow.python.keras.optimizers import nadam_v2, adam_v2, adamax_v2, adagrad_v2
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from data_generator import DataGen
from unet import Unet
from resunet import ResUnet
from resunet_pp import ResUnetPlusPlus
from metrics import dice_coef, dice_loss
from test import gen_dice_coef, gen_dice_loss


if __name__ == "__main__":
    # Start time
    start = time.time()

    # File Paths; Comment out model_path as appropriate
    file_path = "files/"
    # model_path = "files/unet.h5"
    model_path = "files/resunet.h5"
    # model_path = "files/resunetplusplus.h5"

    # Create files folder
    def create_dir(name):
        try:
            os.mkdir("files")
        except:
            pass

    train_path = "new_data/beak_dataset/train/"
    valid_path = "new_data/beak_dataset/valid/"

    # Training
    train_image_paths = glob(os.path.join(train_path, "images", "*"))
    train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
    train_image_paths.sort()
    train_mask_paths.sort()

    # train_image_paths = train_image_paths[:2000]
    # train_mask_paths = train_mask_paths[:2000]

    # Validation
    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_image_paths.sort()
    valid_mask_paths.sort()

    # Parameters
    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 200

    train_steps = len(train_image_paths)//batch_size
    valid_steps = len(valid_image_paths)//batch_size

    # Generator
    train_gen = DataGen(image_size, train_image_paths, train_mask_paths, batch_size=batch_size)
    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)

    # Import model architecture; Comment out as appropriate
    # Unet
    # arch = Unet(input_size=image_size)
    # model = arch.build_model()

    # ResUnet
    # arch = ResUnet(input_size=image_size)
    # model = arch.build_model()

    # ResUnet++
    arch = ResUnetPlusPlus(input_size=image_size)
    model = arch.build_model()

    optimizer = adamax_v2.Adamax(lr)
    metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]
    model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)

    # Set logging, reduced learning on plateau and early stopping
    csv_logger = CSVLogger(f"{file_path}resunet_{batch_size}.csv", append=False)
    checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    callbacks = [csv_logger, checkpoint, reduce_lr, early_stopping]

    # Fit model
    model.fit(x=train_gen,
              validation_data=valid_gen,
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              epochs=epochs,
              callbacks=callbacks)

    # Total time
    end = time.time()
    total_time = end - start
    print("\n"+ str(total_time))
