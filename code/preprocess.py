import os
import random

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import xml.etree.ElementTree as ET
import skimage.io as skio
import skimage.transform as skit

import hyperparameters as hp


class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):

        self.data_path = data_path

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.num_classes

        # Mean and std for standardization
        self.mean = np.zeros((3,))
        self.std = np.zeros((3,))
        self.calc_mean_and_std()

        # Setup data generators
        self.train_data = self.get_data(
            os.path.join(self.data_path, "Data/CLS-LOC/train/"), True, True,
            os.path.join(self.data_path, "Annotations/CLS-LOC/train/"))
        self.test_data = self.get_data(
            os.path.join(self.data_path, "Data/CLS-LOC/val/"), False, False,
            os.path.join(self.data_path, "Annotations/CLS-LOC/val/"))

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".JPEG"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img

        self.mean = np.mean(data_sample, axis=(0, 1, 2))
        self.std = np.std(data_sample, axis=(0, 1, 2))

        print("Dataset mean: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0], self.mean[1], self.mean[2]))

        print("Dataset std: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0], self.std[1], self.std[2]))

    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """

        img = (img - self.mean) / self.std

        return img

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """

        img = img / 255.
        img = self.standardize(img)

        return img

    def get_data(self, path, shuffle, augment, annotation_path):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - the path to the image data
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.
            annotation_path - the path to the annotations of the image data

        Returns:
            An iterable image-batch generator
        """

        if augment:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn,
                rotation_range=35,
                horizontal_flip=True,
                shear_range=10,
                zoom_range=0.5
            )

        else:
            # Don't modify this
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        img_size = hp.img_size

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes  # list of empty strings of length num_classes -> this is probably fine

        # n02085620 -> folder name of chihuahuas
        folder_id = 'n02085620'
        titles_to_annotations = {}

        for direct in os.listdir(annotation_path):
            for annotation in os.listdir(os.path.join(annotation_path, direct)):
                xml_title = annotation[:-4]
                root = ET.parse(os.path.join(annotation_path, os.path.join(direct, annotation)))
                the_annotation = np.zeros(7)
                object_id = root.find('object')
                # name = object_id.find('name')
                bounding_box = object_id.find('bndbox')
                xmin = int(bounding_box.find('xmin').text)
                ymin = int(bounding_box.find('ymin').text)
                xmax = int(bounding_box.find('xmax').text)
                ymax = int(bounding_box.find('ymax').text)
                the_annotation[0] = 1
                the_annotation[1] = xmin
                the_annotation[2] = ymin
                the_annotation[3] = xmax
                the_annotation[4] = ymax
                if annotation == folder_id:
                    the_annotation[5] = 1
                    the_annotation[6] = 0
                else:
                    the_annotation[5] = 0
                    the_annotation[6] = 1
                titles_to_annotations[xml_title] = the_annotation

        annotations = []

        images = []
        for img in os.listdir(path):
            img_title = img[:-5]
            image = skio.imread(os.path.join(path, img))
            image = skit.resize(image, (hp.img_size, hp.img_size))
            images.append(image)
            annotations.append(titles_to_annotations[img_title])
        images = np.asarray(images)
        data_gen = data_gen.flow(images, annotations, batch_size=hp.batch_size)
        # Setup the dictionaries if not already done
        # if not bool(self.idx_to_class):
        #     # this whole thing is built on the names of directories being the labels
        #     #           -> we want it to be built on the object name
        #     # this gets a single xml file's object
        #     #   -> iterate through the whole directory and put these in unordered classes
        #     #       (idk what the file structure looks like)
        #     # root = ET.parse(path) -> this is the path of the label
        #     # object_id = root.find('object').find('name').text
        #     unordered_classes = []
        #     for dir_name in os.listdir(path):
        #         if os.path.isdir(os.path.join(path, dir_name)):
        #             unordered_classes.append(dir_name)  # gets labeled directory names
        #
        #     for img_class in unordered_classes:  # for each image class
        #         self.idx_to_class[data_gen.class_indices[img_class]] = img_class
        #         self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
        #         self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen
