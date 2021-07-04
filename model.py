import tensorflow as tf
from losses import *

def build_model(height, width):
    x_input = tf.keras.Input(shape=(height, width, 3))

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    ########## block 1 ##########
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

    for i in range(2):
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Add()([x_shortcut, x])
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x_shortcut = x

    ########## block 2 ##########
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

    for i in range(2):
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Add()([x_shortcut, x])
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x_shortcut = x

    ########## block 3 ##########
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

    for i in range(8):
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Add()([x_shortcut, x])
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x_shortcut = x

    ########## block 4 ##########
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

    for i in range(8):
        x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Add()([x_shortcut, x])
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x_shortcut = x

    ########## block 5 ##########
    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

    for i in range(4):
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Add()([x_shortcut, x])
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x_shortcut = x

    ########## output layers ##########
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    predictions = tf.keras.layers.Conv2D(10, (1, 1), strides=(1, 1), activation='sigmoid')(x)

    model = tf.keras.Model(inputs=x_input, outputs=predictions)


    return model

def compile_model(model):
    # Compile and Training Model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimiser,
        loss=custom_loss
    )

    return model