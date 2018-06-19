from data_augmentation import arguments
import os
import logging
import tqdm
import cv2
import numpy as np

generator_options = arguments.GeneratorOptions()


def fetch_background_images():
    """
    :return: Returns a list of background images.
    :raises: Warning if background does not have the required dimension.
            The background is then resized.
    """

    backgrounds_path = generator_options.backgrounds_path
    background_files = os.listdir(backgrounds_path)
    background_files = [os.path.join(backgrounds_path, file)
                        for file in background_files]

    background_imgs = list()
    for file in background_files:
        img = cv2.imread(file)
        if list(img.shape[0:2]) != generator_options.image_dimension:
            logging.warning('Background dimension {} not expected'.format(
                        img.shape))
            logging.warning('Rescaling background to shape: {}'.format(
                tuple(generator_options.image_dimension + [3])))
            img = cv2.resize(img, tuple(reversed(
                        generator_options.image_dimension)))

        background_imgs.append(img)

    return background_imgs


def fetch_image_gt_paths():
    """
    This function counts the number of annotated images and
    fetches the path of the images and corresponding labels.

    :return: files_counter: The number of images read.
             object_files_dict: A dictionary which maps object names
                                to corresponding image and label paths.
    """

    object_files_dict = dict()
    files_counter = 0

    for item in os.listdir(generator_options.label_path):
        cls_path = os.path.join(generator_options.label_path, item)
        if os.path.isdir(cls_path):
            obj_files = list()
            for files in sorted(os.listdir(cls_path)):
                files_counter += 1
                obj_files.append([os.path.join(
                            generator_options.image_path, item,
                            files.split('.')[0] + generator_options.real_img_type),
                            os.path.join(generator_options.label_path, item, files)])

                object_files_dict[item] = obj_files.copy()
        else:
            files_counter += 1
            cls_name = '_'.join(item.split('_')[:-1])
            if object_files_dict.get(cls_name, None) is None:
                object_files_dict[cls_name] = list()
            object_files_dict[cls_name].append([os.path.join(
                    generator_options.image_path,
                    item.split('.')[0] + generator_options.real_img_type),
                    os.path.join(generator_options.label_path, item)])

    return files_counter, object_files_dict


def read_image_labels(object_files_dict):
    """

    :param object_files_dict: A dictionary which maps object names
                                to corresponding image and label paths.
    :return: A dictionary which maps object names to corresponding
             image and label data.
    """

    class_name_to_data_dict = {}
    for key in arguments.LABEL_DEF_MATLAB:
        if key is not 'background':
            data_list = object_files_dict[key]
            class_name_to_data_dict[key] = list()
            for data in data_list:
                img = cv2.imread(data[0])
                label = cv2.imread(data[1], 0)
                class_name_to_data_dict[key].append([img, label])

    return class_name_to_data_dict


background_images = fetch_background_images()

files_count, object_files = fetch_image_gt_paths()
class_name_to_data = read_image_labels(object_files)
