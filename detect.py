import tensorflow as tf
import numpy as np
import argparse
from model import *
import slidingwindow
from config import detect_config
import cv2
from PIL import Image
import albumentations as albu
from utils import *



args = argparse.ArgumentParser(description='Process Training model')
args.add_argument('-i','--img_file', type=str, help='images_file', required=True)
argumens = args.parse_args()

config = detect_config
config.rasterpath = argumens.img_file

model = build_model(int(250),int(250))
model = compile_model(model)
model = model.load_weights('Model/object_detection_model.h5')



def compute_windows(numpy_image, patch_size, patch_overlap):

    if patch_overlap > 1:
        raise ValueError("Patch overlap {} must be between 0 - 1".format(patch_overlap))
    windows = slidingwindow.generate(numpy_image,
                                     slidingwindow.DimOrder.HeightWidthChannel,
                                     patch_size, patch_overlap)

    return (windows)


def detect(pixels):
    test_predictions = []
    val_augmentations = albu.Compose([
        albu.CLAHE(p=1),
        albu.ToGray(p=1)
    ])
    aug_result = val_augmentations(image=pixels)
    pixels = np.array(aug_result['image']) / 255
    pixels = np.expand_dims(pixels, axis=0)
    bboxes = model.predict(pixels)
    test_predictions = np.concatenate(bboxes)

    return test_predictions



def detect_tile(image):
    image = cv2.imread(image)
    image = np.asarray(image)

    # Compute sliding window index
    windows = compute_windows(image, config.image_size, config.overlap)
    ratio = (config.image_size / config.resized_size)


    array = np.empty((0, 4), dtype='float')
    for index, window in enumerate(windows):
        # Crop window and predict
        crop = image[windows[index].indices()]

        img = Image.fromarray(crop, 'RGB')
        img = img.resize((config.resized_size, config.resized_size))
        pixels = np.asarray(img)

        hasil = detect(pixels)
        hasil = prediction_to_bbox(hasil, image_grid)

        bbox = non_max_suppression(hasil, top_n=100)

        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, 0:2]

        objects = bbox

        # Convert to VOC Format

        bbox[:, 2] = objects[:, 0] + objects[:, 2]
        bbox[:, 3] = objects[:, 1] + objects[:, 3]

        # transform coordinates to original system
        xmin, ymin, xmax, ymax = windows[index].getRect()
        bbox[:, 0] = bbox[:, 0] + xmin / ratio
        bbox[:, 2] = bbox[:, 2] + xmin / ratio
        bbox[:, 1] = bbox[:, 1] + ymin / ratio
        bbox[:, 3] = bbox[:, 3] + ymin / ratio

        array = np.vstack((array, bbox))


def nms(boxes, overlapThresh):

   if len(boxes) == 0:
      return []
   pick = []
   x1 = boxes[:,0]
   y1 = boxes[:,1]
   x2 = boxes[:,2]
   y2 = boxes[:,3]
   area = (x2 - x1 + 1) * (y2 - y1 + 1)
   idxs = np.argsort(y2)

   while len(idxs) > 0:
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)
      suppress = [last]
      for pos in range(0, last):

         j = idxs[pos]
         xx1 = max(x1[i], x1[j])
         yy1 = max(y1[i], y1[j])
         xx2 = min(x2[i], x2[j])
         yy2 = min(y2[i], y2[j])
         w = max(0, xx2 - xx1 + 1)
         h = max(0, yy2 - yy1 + 1)
         overlap = float(w * h) / area[j]
         if overlap > overlapThresh:
            suppress.append(pos)
      idxs = np.delete(idxs, suppress)

   return boxes[pick]