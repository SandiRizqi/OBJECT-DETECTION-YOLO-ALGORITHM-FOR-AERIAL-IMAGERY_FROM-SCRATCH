import tensorflow as tf
import numpy as np


def prediction_to_bbox(bboxes, image_grid):
    bboxes = bboxes.copy()

    im_width = (image_grid[:, :, 2] * model.output_shape[1])
    im_height = (image_grid[:, :, 3] * model.output_shape[2])

    # descale x,y
    bboxes[:, :, 1] = (bboxes[:, :, 1] * image_grid[:, :, 2]) + image_grid[:, :, 0]
    bboxes[:, :, 2] = (bboxes[:, :, 2] * image_grid[:, :, 3]) + image_grid[:, :, 1]
    bboxes[:, :, 6] = (bboxes[:, :, 6] * image_grid[:, :, 2]) + image_grid[:, :, 0]
    bboxes[:, :, 7] = (bboxes[:, :, 7] * image_grid[:, :, 3]) + image_grid[:, :, 1]

    # descale width,height
    bboxes[:, :, 3] = bboxes[:, :, 3] * im_width
    bboxes[:, :, 4] = bboxes[:, :, 4] * im_height
    bboxes[:, :, 8] = bboxes[:, :, 8] * im_width
    bboxes[:, :, 9] = bboxes[:, :, 9] * im_height

    # centre x,y to top left x,y
    bboxes[:, :, 1] = bboxes[:, :, 1] - (bboxes[:, :, 3] / 2)
    bboxes[:, :, 2] = bboxes[:, :, 2] - (bboxes[:, :, 4] / 2)
    bboxes[:, :, 6] = bboxes[:, :, 6] - (bboxes[:, :, 8] / 2)
    bboxes[:, :, 7] = bboxes[:, :, 7] - (bboxes[:, :, 9] / 2)

    # width,heigth to x_max,y_max
    bboxes[:, :, 3] = bboxes[:, :, 1] + bboxes[:, :, 3]
    bboxes[:, :, 4] = bboxes[:, :, 2] + bboxes[:, :, 4]
    bboxes[:, :, 8] = bboxes[:, :, 6] + bboxes[:, :, 8]
    bboxes[:, :, 9] = bboxes[:, :, 7] + bboxes[:, :, 9]

    return bboxes


def non_max_suppression(predictions, top_n):
    probabilities = np.concatenate((predictions[:, :, 0].flatten(), predictions[:, :, 5].flatten()), axis=None)

    first_anchors = predictions[:, :, 1:5].reshape((model.output_shape[1] * model.output_shape[2], 4))
    second_anchors = predictions[:, :, 6:10].reshape((model.output_shape[1] * model.output_shape[2], 4))

    bboxes = np.concatenate(
        (first_anchors, second_anchors),
        axis=0
    )

    bboxes = switch_x_y(bboxes)
    bboxes, probabilities = select_top(probabilities, bboxes, top_n=top_n)
    bboxes = switch_x_y(bboxes)

    return bboxes


def switch_x_y(bboxes):
    x1 = bboxes[:, 0].copy()
    y1 = bboxes[:, 1].copy()
    x2 = bboxes[:, 2].copy()
    y2 = bboxes[:, 3].copy()

    bboxes[:, 0] = y1
    bboxes[:, 1] = x1
    bboxes[:, 2] = y2
    bboxes[:, 3] = x2

    return bboxes


def select_top(probabilities, boxes, top_n=10):
    top_indices = tf.image.non_max_suppression(
        boxes=boxes,
        scores=probabilities,
        max_output_size=top_n,
        iou_threshold=0.3,
        score_threshold=0.3
    )

    top_indices = top_indices.numpy()

    return boxes[top_indices], probabilities[top_indices]


def process_predictions(predictions, image_ids, image_grid):
    bboxes = {}

    for i, image_id in enumerate(image_ids):
        predictions[i] = prediction_to_bbox(predictions[i], image_grid)
        bboxes[image_id] = non_max_suppression(predictions[i], top_n=100)

        # back to coco shape
        bboxes[image_id][:, 2:4] = bboxes[image_id][:, 2:4] - bboxes[image_id][:, 0:2]

    return bboxes


