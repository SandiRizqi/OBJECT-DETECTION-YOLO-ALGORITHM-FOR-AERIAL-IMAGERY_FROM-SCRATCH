import tensorflow as tf
import argparse
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from tqdm import tqdm
from model import *
from losses import *
import albumentations as albu



args = argparse.ArgumentParser(description='Process Training model')
args.add_argument('-i','--img_dir', type=str, help='images_directory', required=True)
args.add_argument('-m','--model_dir', type=str, help='model_directory', required=True)
args.add_argument('-s','--resized_size', type=int,help='resized_size', required=True)
args.add_argument('-a','--annotations', type=str,help='annotations_file', required=True)
args.add_argument('-e','--epochs', type=int,help='epochs', required=True)
argumens = args.parse_args()


#Create Config
class config:
    annotations_file = argumens.annotations
    image_dir = argumens.img_dir + '/'
    image_size = 1000
    resized_size = argumens.resized_size
    train_ratio = 0.8
    checkpoint = argumens.model_dir + '/'
    saved_model = argumens.model_dir + '/object_detection_model.h5'

#Load annotations file
labels = pd.read_csv(config.annotations_file)
print(labels.head())

#ground labels base on images_id
def group_boxes(group):
    boundaries = group['yolo_bbox'].str.split(',', expand=True)
    boundaries[0] = boundaries[0].str.slice(start=1)
    boundaries[3] = boundaries[3].str.slice(stop=-1)

    return boundaries.values.astype(float)
labels = labels.groupby('image_id').apply(group_boxes)

#spit data to train and val
train_idx = round(len(np.unique(labels.index.values)) * config.train_ratio)
train_image_ids = np.unique(labels.index.values)[0: train_idx]
val_image_ids = np.unique(labels.index.values)[train_idx:]


def load_image(image_id):
    image = Image.open(config.image_dir + image_id)
    image = image.resize((config.resized_size, config.resized_size))

    return np.asarray(image)


#Loading Train data
print("Loading Training data")
train_pixels = {}
train_labels = {}
for image_id in tqdm(train_image_ids):
    train_pixels[image_id] = load_image(image_id)
    train_labels[image_id] = labels[image_id].copy() * (config.resized_size/config.image_size)


#Loading Val data
print("Loading Validation data data")
val_pixels = {}
val_labels = {}
for image_id in tqdm(val_image_ids):
    val_pixels[image_id] = load_image(image_id)
    val_labels[image_id] = labels[image_id].copy() * (config.resized_size/config.image_size)




model = build_model(config.resized_size,config.resized_size)
print(model.summary())


#Create Data Generator

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_ids, image_pixels, labels=None, batch_size=1, shuffle=False, augment=False):
        self.image_ids = image_ids
        self.image_pixels = image_pixels
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

        self.image_grid = self.form_image_grid()


    def form_image_grid(self):
        image_grid = np.zeros((model.output_shape[1], model.output_shape[2], 4))

        # x, y, width, height
        cell = [0, 0, config.resized_size / model.output_shape[1], config.resized_size / model.output_shape[2]]

        for i in range(0, model.output_shape[1]):
            for j in range(0, model.output_shape[2]):
                image_grid[i ,j] = cell

                cell[0] = cell[0] + cell[2]

            cell[0] = 0
            cell[1] = cell[1] + cell[3]

        return image_grid


def __len__(self):
    return int(np.floor(len(self.image_ids) / self.batch_size))


def on_epoch_end(self):
    self.indexes = np.arange(len(self.image_ids))

    if self.shuffle == True:
        np.random.shuffle(self.indexes)

DataGenerator.__len__ = __len__
DataGenerator.on_epoch_end = on_epoch_end

DataGenerator.train_augmentations = albu.Compose([albu.RandomSizedCrop(
    min_max_height=(config.resized_size, config.resized_size),
    height=config.resized_size, width=config.resized_size, p=0.8),
    albu.OneOf([
        albu.Flip(),
        albu.RandomRotate90()], p=1),
    albu.OneOf([
        albu.HueSaturationValue(),
        albu.RandomBrightnessContrast()], p=1),
    albu.OneOf([
        albu.GaussNoise()], p=0.5),
    albu.Cutout(
        num_holes=8,
        max_h_size=16,
        max_w_size=16,
        p=0.5
    ),
    albu.CLAHE(p=1),
    albu.ToGray(p=1),
    ], bbox_params={'format': 'coco', 'label_fields': ['labels']})

DataGenerator.val_augmentations = albu.Compose([
    albu.CLAHE(p=1),
    albu.ToGray(p=1),
])


def __getitem__(self, index):
    indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

    batch_ids = [self.image_ids[i] for i in indexes]

    X, y = self.__data_generation(batch_ids)

    return X, y


def __data_generation(self, batch_ids):
    X, y = [], []

    # Generate data
    for i, image_id in enumerate(batch_ids):
        pixels = self.image_pixels[image_id]
        bboxes = self.labels[image_id]

        if self.augment:
            pixels, bboxes = self.augment_image(pixels, bboxes)
        else:
            pixels = self.contrast_image(pixels)
            bboxes = self.form_label_grid(bboxes)

        X.append(pixels)
        y.append(bboxes)

    return np.array(X), np.array(y)


def augment_image(self, pixels, bboxes):
    bbox_labels = np.ones(len(bboxes))

    aug_result = self.train_augmentations(image=pixels, bboxes=bboxes, labels=bbox_labels)

    bboxes = self.form_label_grid(aug_result['bboxes'])

    return np.array(aug_result['image']) / 255, bboxes


def contrast_image(self, pixels):
    aug_result = self.val_augmentations(image=pixels)
    return np.array(aug_result['image']) / 255


def form_label_grid(self, bboxes):
    label_grid = np.zeros((model.output_shape[1], model.output_shape[2], 10))

    for i in range(0, model.output_shape[1]):
        for j in range(0, model.output_shape[2]):
            cell = self.image_grid[i, j]
            label_grid[i, j] = self.rect_intersect(cell, bboxes)

    return label_grid


def rect_intersect(self, cell, bboxes):
    cell_x, cell_y, cell_width, cell_height = cell
    cell_x_max = cell_x + cell_width
    cell_y_max = cell_y + cell_height

    anchor_one = np.array([0, 0, 0, 0, 0])
    anchor_two = np.array([0, 0, 0, 0, 0])

    # check all boxes
    for bbox in bboxes:
        box_x, box_y, box_width, box_height = bbox
        box_x_centre = box_x + (box_width / 2)
        box_y_centre = box_y + (box_height / 2)

        if (box_x_centre >= cell_x and box_x_centre < cell_x_max and box_y_centre >= cell_y and box_y_centre < cell_y_max):

            if anchor_one[0] == 0:
                anchor_one = self.yolo_shape(
                    [box_x, box_y, box_width, box_height],
                    [cell_x, cell_y, cell_width, cell_height]
                )

            elif anchor_two[0] == 0:
                anchor_two = self.yolo_shape(
                    [box_x, box_y, box_width, box_height],
                    [cell_x, cell_y, cell_width, cell_height]
                )

            else:
                break

    return np.concatenate((anchor_one, anchor_two), axis=None)


def yolo_shape(self, box, cell):
    box_x, box_y, box_width, box_height = box
    cell_x, cell_y, cell_width, cell_height = cell

    # top left x,y to centre x,y
    box_x = box_x + (box_width / 2)
    box_y = box_y + (box_height / 2)

    # offset bbox x,y to cell x,y
    box_x = (box_x - cell_x) / cell_width
    box_y = (box_y - cell_y) / cell_height

    # bbox width,height relative to cell width,height
    box_width = box_width / config.resized_size
    box_height = box_height / config.resized_size

    return [1, box_x, box_y, box_width, box_height]

#Setting up DataGenerator
DataGenerator.augment_image = augment_image
DataGenerator.contrast_image = contrast_image
DataGenerator.form_label_grid = form_label_grid
DataGenerator.rect_intersect = rect_intersect
DataGenerator.yolo_shape = yolo_shape
DataGenerator.__getitem__ = __getitem__
DataGenerator.__data_generation = __data_generation


train_generator = DataGenerator(
    train_image_ids,
    train_pixels,
    train_labels,
    batch_size=1,
    shuffle=True,
    augment=True
)

val_generator = DataGenerator(
    val_image_ids,
    val_pixels,
    val_labels,
    batch_size=1,
    shuffle=False,
    augment=False
)
image_grid = train_generator.image_grid




#Compile and Training Model
optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimiser,
    loss=custom_loss
)
callbacks = [tf.keras.callbacks.ModelCheckpoint(config.checkpoint + '/object_detection_ckpt.weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, mode='auto', save_weights_only=True), \
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=3, verbose=1), \
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True), \
]

history = model.fit(train_generator,validation_data=val_generator, epochs=argumens.epochs, callbacks=callbacks)
model.save(config.saved_model)