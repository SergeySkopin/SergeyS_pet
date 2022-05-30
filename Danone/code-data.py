#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from PIL import Image
from os import makedirs
from os import path
import datetime
import subprocess
import sys
from loader import get_data_generator
from loader import load_dict_from_file
from loader import load_alphabet
from loader import get_test_generator
from loader import remove_char
import numpy as np


def VGG16_FC(input_shape, weights_path=None, filters=1000, with_detector=True):
    """
    # Arguments
    input_shape: shape tuple
    weights_path: optional path to weights file
    filters: optional number of filters to segment images
    with_detector: add detector head
    # Returns
    A Keras model instance.
    # Raises
    invalid input shape.
    """

    # Determine proper input shape
    # input_shape = _obtain_input_shape(
    # input_shape,
    # default_size=224,
    # min_size=48,
    # data_format=K.image_data_format(),
    # require_flatten=include_top,
    # weights=weights)

    img_input = Input(shape=input_shape)

    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    # Block 1
    x1 = Conv2D(16, (3, 3), padding="same", name="block1_conv1")(img_input)
    x1 = BatchNormalization(axis=bn_axis, name="block1_bn1")(x1)
    x1 = Activation("relu")(x1)

    x1 = Conv2D(32, (3, 3), padding="same", name="block1_conv2")(x1)
    x1 = BatchNormalization(axis=bn_axis, name="block1_bn2")(x1)
    x1 = Activation("relu")(x1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x1)

    # Block 2
    x2 = Conv2D(64, (3, 3), padding="same", name="block2_conv1")(pool1)
    x2 = BatchNormalization(axis=bn_axis, name="block2_bn1")(x2)
    x2 = Activation("relu")(x2)

    x2 = Conv2D(64, (3, 3), padding="same", name="block2_conv2")(x2)
    x2 = BatchNormalization(axis=bn_axis, name="block2_bn2")(x2)
    x2 = Activation("relu")(x2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x2)

    # Block 3
    x3 = Conv2D(64, (3, 3), padding="same", name="block3_conv1")(pool2)
    x3 = BatchNormalization(axis=bn_axis, name="block3_bn1")(x3)
    x3 = Activation("relu")(x3)

    x3 = Conv2D(64, (3, 3), padding="same", name="block3_conv2")(x3)
    x3 = BatchNormalization(axis=bn_axis, name="block3_bn2")(x3)
    x3 = Activation("relu")(x3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x3)

    # Block 4
    x4 = Conv2D(128, (3, 3), padding="same", name="block4_conv1")(pool3)
    x4 = BatchNormalization(axis=bn_axis, name="block4_bn1")(x4)
    x4 = Activation("relu")(x4)

    x4 = Conv2D(128, (3, 3), padding="same", name="block4_conv2")(x4)
    x4 = BatchNormalization(axis=bn_axis, name="block4_bn2")(x4)
    x4 = Activation("relu")(x4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x4)

    # Block 5
    x5 = Conv2D(128, (3, 3), padding="same", name="block5_conv1")(pool4)
    x5 = BatchNormalization(axis=bn_axis, name="block5_bn1")(x5)
    x5 = Activation("relu")(x5)

    x5 = Conv2D(128, (3, 3), padding="same", name="block5_conv2")(x5)
    x5 = BatchNormalization(axis=bn_axis, name="block5_bn2")(x5)
    x5 = Activation("relu")(x5)

    # Up Block 6
    x6 = concatenate(
        [
            Conv2DTranspose(
                128, (2, 2), strides=(2, 2), padding="same", name="block6_transpose"
            )(x5),
            x4,
        ],
        axis=3,
    )

    x6 = Conv2D(128, (3, 3), padding="same", name="block6_conv1")(x6)
    x6 = BatchNormalization(axis=bn_axis, name="block6_bn1")(x6)
    x6 = Activation("relu")(x6)

    x6 = Conv2D(64, (3, 3), padding="same", name="block6_conv2")(x6)
    x6 = BatchNormalization(axis=bn_axis, name="block6_bn2")(x6)
    x6 = Activation("relu")(x6)

    x6 = Conv2D(64, (3, 3), padding="same", name="block6_conv3")(x6)
    x6 = BatchNormalization(axis=bn_axis, name="block6_bn3")(x6)
    x6 = Activation("relu")(x6)

    # Segmentation output
    segmentation_output = Conv2D(
        filters, (3, 3), activation="sigmoid", padding="same", name="out"
    )(x6)

    if not with_detector:
        model = Model(
            inputs=img_input, outputs=segmentation_output, name="vgg16_mutant"
        )
    else:
        # Coords
        # Aux block 1
        x = Conv2D(64, (3, 3), padding="same", name="aux_block1_conv1")(x6)
        x = BatchNormalization(axis=bn_axis, name="aux_block1_bn1")(x)
        x = Activation("relu")(x)

        x = Conv2D(64, (3, 3), padding="same", name="aux_block1_conv2")(x)
        x = BatchNormalization(axis=bn_axis, name="aux_block1_bn2")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="aux_block1_pool")(x)

        # Aux block 2
        x = Conv2D(64, (3, 3), padding="same", name="aux_block2_conv1")(x)
        x = BatchNormalization(axis=bn_axis, name="aux_block2_bn1")(x)
        x = Activation("relu")(x)

        x = Conv2D(64, (3, 3), padding="same", name="aux_block2_conv2")(x)
        x = BatchNormalization(axis=bn_axis, name="aux_block2_bn2")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="aux_block2_pool")(x)

        # Aux block 3
        x = Conv2D(64, (3, 3), padding="same", name="aux_block3_conv1")(x)
        x = BatchNormalization(axis=bn_axis, name="aux_block3_bn1")(x)
        x = Activation("relu")(x)

        x = Conv2D(64, (3, 3), padding="same", name="aux_block3_conv2")(x)
        x = BatchNormalization(axis=bn_axis, name="aux_block3_bn2")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="aux_block3_pool")(x)

        # Aux block 4
        x = Conv2D(128, (3, 3), padding="same", name="aux_block4_conv1")(x)
        x = BatchNormalization(axis=bn_axis, name="aux_block4_bn1")(x)
        x = Activation("relu")(x)

        x = Conv2D(128, (3, 3), padding="same", name="aux_block4_conv2")(x)
        x = BatchNormalization(axis=bn_axis, name="aux_block4_bn2")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="aux_block4_pool")(x)

        # Aux block 5
        x = Conv2D(128, (3, 3), padding="same", name="aux_block5_conv1")(x)
        x = BatchNormalization(axis=bn_axis, name="aux_block5_bn1")(x)
        x = Activation("relu")(x)

        x = Conv2D(128, (3, 3), padding="same", name="aux_block5_conv2")(x)
        x = BatchNormalization(axis=bn_axis, name="aux_block5_bn2")(x)
        x = Activation("relu")(x)

        # Aux block 6
        x = Conv2D(128, (3, 3), padding="same", name="aux_block6_conv1")(x)
        x = BatchNormalization(axis=bn_axis, name="aux_block6_bn1")(x)
        x = Activation("relu")(x)

        x = Conv2D(128, (3, 3), padding="same", name="aux_block6_conv2")(x)
        x = BatchNormalization(axis=bn_axis, name="aux_block6_bn2")(x)
        x = Activation("relu")(x)
        x = Flatten()(x)

        # Coords output
        coords_output = Dense(8, activation="sigmoid", name="aux_output")(x)

        # Create model
        model = Model(
            inputs=img_input,
            outputs=[segmentation_output, coords_output],
            name="vgg16_mutant",
        )

    # load weights
    if type(weights_path) is str:
        model.load_weights(weights_path, by_name=True)
    elif type(weights_path) is list:
        for p in weights_path:
            model.load_weights(p, by_name=True)

    return model


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()


def speed_benchmark(model, batch_size, iterations, run_date):
    shape = model.layers[0].get_config()["batch_input_shape"]
    batch_shape = (batch_size,) + shape[1:]
    input_array = np.random.rand(*batch_shape)
    t0 = datetime.datetime.now()
    for i in range(iterations):
        model.predict(input_array, batch_size=batch_size)
    t1 = datetime.datetime.now()
    result = (t1 - t0) / (iterations * batch_size)

    log_file = "speed_benchmark.txt"
    with open(log_file, "a") as f:
        f.write("{}, {}\n".format(run_date, result))


def freeze_segmentation_leyers(model):
    for layer in model.layers:
        if "aux_" not in layer.name:
            layer.trainable = False
    return model


smooth = 1.0


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def main():
    action = sys.argv[1]
    run_date = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    run_date = run_date + "_commit_" + get_git_revision_hash()
    print("=================================================================")
    print(run_date)
    print("=================================================================")
    epoch_path = "epoch"
    filename_temp = path.join(
        epoch_path, "epochs-run_{}".format(run_date) + "-{epoch:04d}.hdf5"
    )
    if action == "train_all":
        batch_size = 10
    elif action == "train_segmentation":
        batch_size = 8
    elif action == "train_detector":
        batch_size = 16
    else:
        batch_size = 16

    if action == "train_segmentation":
        train_samples_cnt = 16000
    else:
        train_samples_cnt = 16000

    test_samples_cnt = 512
    epochs = 100

    if not path.exists(epoch_path):
        makedirs(epoch_path)

    if K.image_data_format() == "channels_first":
        if K.backend() == "tensorflow":
            warnings.warn(
                "You are using the TensorFlow backend, yet you "
                "are using the Theano "
                "image data format convention "
                '(`image_data_format="channels_first"`). '
                "For best performance, set "
                '`image_data_format="channels_last"` in '
                "your Keras config "
                "at ~/.keras/keras.json."
            )

    alphabet = load_alphabet("../augmentation/alphabet.txt")
    alphabet = remove_char(alphabet, " ")

    checkpoint = ModelCheckpoint(filepath=filename_temp, save_weights_only=True)

    if action == "speed_benchmark":
        input_shape = (512, 1280, 1)  # (h, w, ch)
        with_detector = True
    elif action == "validation_segmentation":
        input_shape = (512, 1280, 1)  # (h, w, ch)
        with_detector = False
    elif action == "test_segmentation":
        input_shape = (352, 1056, 1)  # (h, w, ch)
        with_detector = False
    elif action == "validation_detector":
        input_shape = (512, 1280, 1)  # (h, w, ch)
        with_detector = True
    elif action == "test_detector":
        input_shape = (512, 1280, 1)  # (h, w, ch)
        with_detector = True
    elif action == "train_segmentation":
        input_shape = (352, 1056, 1)  # (h, w, ch)
        with_detector = False
    elif action == "train_detector":
        input_shape = (512, 1280, 1)  # (h, w, ch)
        with_detector = True
    elif action == "train_all":
        input_shape = (512, 1280, 1)  # (h, w, ch)
        with_detector = True

    if len(sys.argv) == 3:
        weights_path = sys.argv[2]
    elif len(sys.argv) == 4:
        weights_path = [sys.argv[2], sys.argv[3]]
    else:
        weights_path = None

    model = VGG16_FC(
        input_shape=input_shape,
        weights_path=weights_path,
        filters=len(alphabet),
        with_detector=with_detector,
    )

    if action == "train_detector":
        model = freeze_segmentation_leyers(model)

    if action == "train_detector":
        loss_weights = {"out": 0, "aux_output": 1}
    elif action == "train_all":
        loss_weights = {"out": 1e-3, "aux_output": 1}
    else:
        loss_weights = {"out": 1, "aux_output": 1}

    optimizer = Adam()  # SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    if with_detector:
        model.compile(
            optimizer=optimizer,
            loss={"out": "binary_crossentropy", "aux_output": "mean_squared_error"},
            metrics={"out": ["accuracy", dice_coef], "aux_output": []},
            loss_weights=loss_weights,
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", dice_coef, precision, recall],
        )

    if action == "validation_segmentation":
        test_generator = get_test_generator(
            alphabet,
            "./ora/",
            batch_size=1,
            input_shape=input_shape,
            pooling_koef=8,
            with_detector=with_detector,
        )
    elif action == "test_segmentation":
        test_data = load_dict_from_file("../augmentation/test_segmentation_log.json")
        test_generator = get_data_generator(
            alphabet,
            test_data,
            "../augmentation",
            batch_size=batch_size,
            input_shape=input_shape,
            pooling_koef=8,
            with_detector=with_detector,
        )
    elif action == "validation_detector":
        test_generator = get_test_generator(
            alphabet,
            "./ora/",
            batch_size=1,
            input_shape=input_shape,
            pooling_koef=8,
            with_detector=with_detector,
        )
    elif action == "test_detector":
        test_data = load_dict_from_file("../augmentation/test_log.json")
        test_generator = get_data_generator(
            alphabet,
            test_data,
            "../augmentation",
            batch_size=batch_size,
            input_shape=input_shape,
            pooling_koef=8,
            with_detector=with_detector,
        )
    elif action == "train_segmentation":
        train_data = load_dict_from_file("../augmentation/train_segmentation_log.json")
        train_generator = get_data_generator(
            alphabet,
            train_data,
            "../augmentation/",
            batch_size=batch_size,
            input_shape=input_shape,
            pooling_koef=8,
            with_detector=with_detector,
        )

        test_data = load_dict_from_file("../augmentation/test_segmentation_log.json")
        test_generator = get_data_generator(
            alphabet,
            test_data,
            "../augmentation",
            batch_size=batch_size,
            input_shape=input_shape,
            pooling_koef=8,
            with_detector=with_detector,
        )

    elif action == "train_detector" or action == "train_all":
        train_data = load_dict_from_file("../augmentation/train_log.json")
        train_generator = get_data_generator(
            alphabet,
            train_data,
            "../augmentation/",
            batch_size=batch_size,
            input_shape=input_shape,
            pooling_koef=8,
            with_detector=with_detector,
        )

        test_data = load_dict_from_file("../augmentation/test_log.json")
        test_generator = get_data_generator(
            alphabet,
            test_data,
            "../augmentation",
            batch_size=batch_size,
            input_shape=input_shape,
            pooling_koef=8,
            with_detector=with_detector,
        )

    if action == "speed_benchmark":
        speed_benchmark(model, 1, 1000, run_date)

    elif action == "validation_segmentation":
        result = model.evaluate_generator(test_generator, 6)
        print(result)

        log_file = path.splitext(weights_path)[0] + "_validation_segmentation.txt"
        with open(log_file, "w") as f:
            f.write("Validation, run: {}, weights: {}, ".format(run_date, weights_path))
            f.write(
                "loss: {}, accuracy: {}, dice: {}, precision: {}, recall: {}\n".format(
                    result[0], result[1], result[2], result[3], result[4]
                )
            )
    elif action == "test_segmentation":
        result = model.evaluate_generator(test_generator, 512)
        print(result)

        log_file = path.splitext(weights_path)[0] + "_test_segmentation.txt"
        with open(log_file, "w") as f:
            f.write("Test, run: {}, weights: {}, ".format(run_date, weights_path))
            f.write(
                "loss: {}, accuracy: {}, dice: {}\n".format(
                    result[0], result[1], result[2]
                )
            )
    elif action == "validation_detector":
        result = model.evaluate_generator(test_generator, 6)
        print(result)

        log_file = path.splitext(weights_path)[0] + "_validation_detector.txt"
        with open(log_file, "w") as f:
            f.write("Validation, run: {}, weights: {}, ".format(run_date, weights_path))
            f.write("loss: {}\n".format(result[2]))
    elif action == "test_detector":
        result = model.evaluate_generator(test_generator, 512)
        print(result)

        log_file = path.splitext(weights_path)[0] + "_test_detector.txt"
        with open(log_file, "w") as f:
            f.write("Test, run: {}, weights: {}, ".format(run_date, weights_path))
            f.write("loss: {}\n".format(result[2]))

    elif (
        action == "train_segmentation"
        or action == "train_detector"
        or action == "train_all"
    ):

        checkpoint = ModelCheckpoint(filepath=filename_temp, save_weights_only=True)

        tensorboard = TensorBoard(
            log_dir="./logs_{}".format(run_date),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
        )

        model.summary()

        model.fit_generator(
            train_generator,
            validation_data=test_generator,
            validation_steps=test_samples_cnt // batch_size,
            steps_per_epoch=train_samples_cnt // batch_size,
            epochs=epochs,
            initial_epoch=0,
            callbacks=[checkpoint, tensorboard],
        )


if __name__ == "__main__":
    main()
