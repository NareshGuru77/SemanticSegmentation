from data_augmentation.arguments import generator_options, LABEL_DEF_MATLAB
from data_augmentation.object_details import objects
from data_augmentation.generate_augmenter_list import augmenter_list
from data_augmentation.visualizer import save_visuals
import copy
import numpy as np
import tqdm
import cv2
import os
import csv
from joblib import Parallel, delayed
import multiprocessing


def get_locations_in_image(obj_locations):
    """
    :param obj_locations: List of (x,y) locations.
    :return: array of locations within the image space.
    """
    locs_within_image = []
    for index, location in enumerate(obj_locations):
        if (0 <= location[0] <= generator_options.image_dimension[0]
                and 0 <= location[1] <= generator_options.image_dimension[1]):
            locs_within_image.append(location)

    return np.array(locs_within_image)


def get_augmented_image(original_image, original_label,
                        obj_details, location):
    """
    This function gets an image, label and object details and returns
    a new image and label with the object placed.

    :param original_image: The image on which an object needs to be placed.
    :param original_label: The corresponding label image.
    :param obj_details: The details dictionary of the object to be placed.
    :param location: The location in pixel coordinates where the object
                     needs to be placed.
    :return: returns image and label augmented with the object to be placed.
    """

    augmented_image = original_image.copy()
    augmented_label = original_label.copy()
    obj_details_to_augment = copy.deepcopy(obj_details)

    min_loc_index = np.argmin(np.sum(
        obj_details_to_augment['obj_loc'], axis=1))
    obj_details_to_augment['obj_loc'] -= (obj_details_to_augment['obj_loc'][
                                          min_loc_index, :] - location)

    for index, loc in enumerate(obj_details_to_augment['obj_loc']):
        if (0 < loc[0] < generator_options.image_dimension[0]
                and 0 < loc[1] < generator_options.image_dimension[1]):
            augmented_image[tuple(loc)] = obj_details_to_augment[
                                    'obj_vals'][index]
            augmented_label[tuple(loc)] = obj_details_to_augment[
                                    'label_vals'][index]

    if generator_options.save_obj_det_label:
        obj_locations = get_locations_in_image(
                                obj_details_to_augment['obj_loc'])
        rect_points = [min(obj_locations[:, 0]), min(obj_locations[:, 1]),
                       max(obj_locations[:, 0]), max(obj_locations[:, 1])]

        obj_det_label = [obj_details_to_augment['obj_name']] + rect_points
        return augmented_image, augmented_label, obj_det_label

    return augmented_image, augmented_label


def make_save_dirs():
    """
    This function checks whether the save paths exists. Creates them if
    they do not exist.
    :return: No returns.
    """

    if not os.path.isdir(generator_options.image_save_path):
        os.makedirs(generator_options.image_save_path)

    if not os.path.isdir(generator_options.label_save_path):
        os.makedirs(generator_options.label_save_path)

    if generator_options.save_obj_det_label:
        if not os.path.isdir(generator_options.obj_det_save_path):
            os.makedirs(generator_options.obj_det_save_path)

    if generator_options.save_mask:
        if not os.path.isdir(generator_options.mask_save_path):
            os.makedirs(generator_options.mask_save_path)

    if generator_options.save_label_preview:
        if not os.path.isdir(generator_options.preview_save_path):
            os.makedirs(generator_options.preview_save_path)

    if generator_options.save_overlay:
        if not os.path.isdir(generator_options.overlay_save_path):
            os.makedirs(generator_options.overlay_save_path)


def save_data(artificial_image, semantic_label, obj_det_label, index):
    """
    This function saves the artificial image and its corresponding semantic
    label. Also saves object detection labels, plot preview and segmentation
    mask images based on "generator_options".

    :param artificial_image: The artificial image which needs to be saved.
    :param semantic_label: The semantic segmentation label image which
                           needs to be saved.
    :param obj_det_label: The object detection label which needs to be
                          saved. Can be None if "save_obj_det_label" is false.
    :param index: The index value to be included in the name of the files.
    :return: No returns.
    """

    cv2.imwrite(os.path.join(
        generator_options.image_save_path,
        generator_options.name_format %
        (index + generator_options.start_index) + '.jpg'),
        artificial_image)

    cv2.imwrite(os.path.join(
        generator_options.label_save_path,
        generator_options.name_format %
        (index + generator_options.start_index) + '.png'),
        semantic_label)

    if generator_options.save_obj_det_label:
        with open(os.path.join(
                generator_options.obj_det_save_path,
                generator_options.name_format %
                (index + generator_options.start_index) + '.csv'), 'w') as f:

            wr = csv.writer(f, delimiter=',')
            [wr.writerow(l) for l in obj_det_label]
    else:
        obj_det_label = None

    if (generator_options.save_mask or
            generator_options.save_label_preview or
            generator_options.save_overlay):
        save_visuals(artificial_image, semantic_label,
                     obj_det_label, index)


def worker(index, element, obj_det_label, background_label):
    """
    This is a worker function created for parallel processing
     of "perform_augmentation" function.
    :param index: The index of the current element.
    :param element: The current element in the augment vector.
    :param obj_det_label: Object detection label.
    :param background_label: A 1 channel image filled with
                             background label value.
    :return: No returns.
    """
    artificial_image = element['background_image']
    semantic_label = background_label.copy()
    obj_det_label.clear()
    obj_details_list = [objects[this_object]
                        for this_object in element['what_objects']]
    obj_details_list = sorted(obj_details_list,
                              key=lambda k: k['obj_area'],
                              reverse=True)
    for i in range(element['num_objects_to_place']):

        if generator_options.save_obj_det_label:
            artificial_image, semantic_label, rect_label = (
                get_augmented_image(artificial_image,
                                    semantic_label,
                                    obj_details_list[i],
                                    element['locations'][i]))
            obj_det_label.append(rect_label)
        else:
            artificial_image, semantic_label = (
                get_augmented_image(artificial_image,
                                    semantic_label,
                                    obj_details_list[i],
                                    element['locations'][i]))

    save_data(artificial_image, semantic_label, obj_det_label, index)


def perform_augmentation():
    """

    This function goes through the augmenter list and generates an artificial
    image for each element in the augmenter list. The results are saved in the
    corresponding locations specified by "generator_options".

    In each element of augmenter list, objects in 'what_objects' is taken and
    pasted on top of the 'background_image' in the element...
    :return: No returns.
    """

    make_save_dirs()
    obj_det_label = list()
    background_label = np.ones(tuple(
        generator_options.image_dimension)) * (
        LABEL_DEF_MATLAB['background'])

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(worker)(index,
                                               element, obj_det_label, background_label)
                               for index, element in enumerate(tqdm.tqdm(
                                                    augmenter_list,
                                                    desc='Generating artificial images')))


if __name__ == '__main__':
    perform_augmentation()
