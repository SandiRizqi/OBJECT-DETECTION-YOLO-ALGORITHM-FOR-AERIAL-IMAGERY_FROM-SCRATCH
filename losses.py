import tensorflow as tf

def custom_loss(y_true, y_pred):
    binary_crossentropy = prob_loss = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )

    prob_loss = binary_crossentropy(
        tf.concat([y_true[:, :, :, 0], y_true[:, :, :, 5]], axis=0),
        tf.concat([y_pred[:, :, :, 0], y_pred[:, :, :, 5]], axis=0)
    )

    xy_loss = tf.keras.losses.MSE(
        tf.concat([y_true[:, :, :, 1:3], y_true[:, :, :, 6:8]], axis=0),
        tf.concat([y_pred[:, :, :, 1:3], y_pred[:, :, :, 6:8]], axis=0)
    )

    wh_loss = tf.keras.losses.MSE(
        tf.concat([y_true[:, :, :, 3:5], y_true[:, :, :, 8:10]], axis=0),
        tf.concat([y_pred[:, :, :, 3:5], y_pred[:, :, :, 8:10]], axis=0)
    )

    bboxes_mask = get_mask(y_true)

    xy_loss = xy_loss * bboxes_mask
    wh_loss = wh_loss * bboxes_mask

    return prob_loss + xy_loss + wh_loss


def get_mask(y_true):
    anchor_one_mask = tf.where(
        y_true[:, :, :, 0] == 0,
        0.5,
        5.0
    )

    anchor_two_mask = tf.where(
        y_true[:, :, :, 5] == 0,
        0.5,
        5.0
    )

    bboxes_mask = tf.concat(
        [anchor_one_mask, anchor_two_mask],
        axis=0
    )

    return bboxes_mask