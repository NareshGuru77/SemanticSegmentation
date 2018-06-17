from data_augmentation import arguments
import os
import logging
import tqdm
import cv2
import numpy as np

generator_options = arguments.GeneratorOptions()


def fetch_background_images():
    """
    This function reads the background images and creates a list...
    """
    backgrounds_path = generator_options.backgrounds_path
    background_files = os.listdir(backgrounds_path)
    background_files = [os.path.join(backgrounds_path, file)
                        for file in background_files]

    background_images = list()
    for file in background_files:
        img = cv2.imread(file)
        if list(img.shape[0:2]) != generator_options.image_dimension:
            logging.warning('Background dimension {} not expected'.format(
                        img.shape))
            logging.warning('Rescaling background to shape: {}'.format(
                tuple(generator_options.image_dimension + [3])))
            img = cv2.resize(img, tuple(reversed(
                        generator_options.image_dimension)))

        background_images.append(img)

    return background_images


def fetch_image_gt_paths():
    """
    This function counts the number of annotated images and
    fetches the path of the images and corresponding labels..

    Returns the number of annotated images and a dictionary mapping
    the object name to the corresponding image and label paths...
    """
    object_files = dict()
    files_count = 0

    for item in os.listdir(generator_options.label_path):
        cls_path = os.path.join(generator_options.label_path, item)
        if os.path.isdir(cls_path):
            obj_files = list()
            for files in sorted(os.listdir(cls_path)):
                files_count += 1
                obj_files.append([os.path.join(
                            generator_options.image_path, item,
                            files.split('.')[0] + generator_options.real_img_type),
                            os.path.join(generator_options.label_path, item, files)])

                object_files[item] = obj_files.copy()
        else:
            files_count += 1
            cls_name = '_'.join(item.split('_')[:-1])
            if object_files.get(cls_name, None) is None:
                object_files[cls_name] = list()
            object_files[cls_name].append([os.path.join(
                    generator_options.image_path,
                    item.split('.')[0] + generator_options.real_img_type),
                    os.path.join(generator_options.label_path, item)])

    return files_count, object_files


def read_image_labels(object_files):

    class_name_to_data = {}
    for key in arguments.LABEL_DEF_MATLAB:
        if key is not 'background':
            data_list = object_files[key]
            class_name_to_data[key] = list()
            for data in data_list:
                img = cv2.imread(data[0])
                label = cv2.imread(data[1], 0)
                class_name_to_data[key].append([img, label])

    return class_name_to_data

background_images = fetch_background_images()

files_count, object_files = fetch_image_gt_paths()
class_name_to_data =  read_image_labels(object_files)
