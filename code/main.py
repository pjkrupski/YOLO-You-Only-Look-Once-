import argparse
from datetime import datetime
import os
import tensorflow as tf
import re
import sys

import hyperparameters as hp

from yolo_model import YoloModel
from preprocess import Datasets
from tensorboard_utils import ImageLabelingLogger, CustomModelSaver


def parse_args():
    parser = argparse.ArgumentParser(
        description="for training our YOLO model!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data',
        default='/mnt/disks/extra_disk/ILSVRC',
        help='''The path to where the data is stored'''
    )
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to the checkpoint file to load into the model'''
    )
    parser.add_argument(
        '--evaluate',
        default=None,
        action='store_true',
        help='''Please enter the weights that will be loaded into the model 
                to perform visualizable results.'''
    )

    return parser.parse_args()


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0
        ),
        #ImageLabelingLogger(logs_path, datasets),
        #CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
    ]

    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=hp.batch_size,
        callbacks=callback_list,
        initial_epoch=init_epoch
    )


def test(model, test_data):
    model.evaluate(
        x=test_data,
        verbose=1
    )


def main():
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # loading a checkpoint
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)

    os.chdir(sys.path[0])

    print(ARGS.data)
    #ARGS.data = '/mnt/disks/extra_disk/ILSVRC'
    datasets = Datasets(ARGS.data)  # ARGS.data will be the absolute path to the data

    model = YoloModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    checkpoint_path = "checkpoints" + os.sep + "yolo_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "yolo_model" + os.sep + timestamp + os.sep
    model.summary()

    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint, by_name=False)

    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"]
    )

    if ARGS.evaluate:
        test(model, datasets.test_data)
        # path = ARGS.data + os.sep + ARGS.image_name
        # visualize(model, path, datasets.preprocess_fn)  # TODO: create a visualizing function
    else:
        train(model, datasets, checkpoint_path, logs_path, init_epoch)


ARGS = parse_args()

if __name__ == "__main__":
    main()
