"""
Inference testing for ResUNet++, ResUNet or UNet models.
 - Inference based on images from testing in new_data/dataset
 - Model weights are read from files directory
"""

import os
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.utils import CustomObjectScope
from tensorflow.python.keras.utils.all_utils import CustomObjectScope
from data_generator import *
from metrics import dice_coef, dice_loss
from tensorflow.python.keras.models import Model
from unet import Unet
from resunet import ResUnet
from resunet_pp import ResUnetPlusPlus
from tensorflow.python.keras.metrics import (Precision, Recall, MeanIoU)
from tensorflow.python.keras.optimizers import nadam_v2, adam_v2, adamax_v2, adagrad_v2

# Model parameters (must match those in run.py, on training)
lr = 1e-4


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


if __name__ == "__main__":
    # Start time
    start = time.time()

    # File Paths; Comment out model_path as appropriate
    # model_path = "files/unet.h5"
    # model_path = "files/resunet.h5"
    model_path = "files/resunetplusplus.h5"
    save_path = "result"
    test_path = "new_data/beak_dataset/test/"

    # Model parameters
    image_size = 256
    batch_size = 1

    test_image_paths = glob(os.path.join(test_path, "images", "*"))
    test_mask_paths = glob(os.path.join(test_path, "masks", "*"))
    test_image_paths.sort()
    test_mask_paths.sort()

    # Create result folder
    try:
        os.mkdir(save_path)
    except:
        pass


    # Rebuild model with weights from training
    with ((CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}))):

        # Build Model; Comment out model as appropriate
        # arch = Unet(input_size=image_size)
        # arch = ResUnet(input_size=image_size)
        arch = ResUnetPlusPlus(input_size=image_size)
        model = arch.build_model()

        optimizer = adamax_v2.Adamax(lr)
        metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]
        model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)

        # Load model weights; Comment out model as appropriate
        # model.load_weights("files/unet.h5")
        # model.load_weights("files/resunet.h5")
        model.load_weights("files/resunetplusplus.h5")


    # Test
    print("Test Result: ")
    test_steps = len(test_image_paths)//batch_size
    test_gen = DataGen(image_size, test_image_paths, test_mask_paths, batch_size=batch_size)
    model.evaluate(test_gen, steps=test_steps, verbose=1)

    # Generating the result
    for i, path in tqdm(enumerate(test_image_paths), total=len(test_image_paths)):
        image = parse_image(test_image_paths[i], image_size)
        mask = parse_mask(test_mask_paths[i], image_size)

        predict_mask = model.predict(np.expand_dims(image, axis=0))[0]
        predict_mask = (predict_mask > 0.5) * 255.0

        sep_line = np.ones((image_size, 10, 3)) * 255

        mask = mask_to_3d(mask)
        predict_mask = mask_to_3d(predict_mask)

        all_images = [image * 255, sep_line, mask * 255, sep_line, predict_mask]
        cv2.imwrite(f"{save_path}/{i}.png", np.concatenate(all_images, axis=1))

    print("Test image generation complete")

    # Total time
    end = time.time()
    total_time = end - start
    print("\n"+ str(total_time))
