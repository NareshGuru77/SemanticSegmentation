from data_augmentation import arguments
from data_augmentation import object_details
from data_augmentation import generate_augmenter_list
from data_augmentation import plotter
import copy
import numpy as np
import tqdm
import cv2
import os
import csv

generator_options = arguments.GeneratorOptions()


def get_augmented_image(original_image, original_label,
                        obj_vals, location):
    """
    This function gets the image and label which needs
    to be augmented, the object details
    and the random location in which the object needs
    to be placed as arguments..

    The existing object location is shifted based
    on the location argument and the
    intensity values of the object are now placed
    in the shifted location...

    The resultant augmented image and label is returned...
    """
    augmented_image = original_image.copy()
    augmented_label = original_label.copy()
    obj_vals_to_augment = copy.deepcopy(obj_vals)

    row_shift = min(obj_vals_to_augment['obj_loc'][:, 0]
                    ) - location[0]
    col_shift = min(obj_vals_to_augment['obj_loc'][:, 1]
                    ) - location[1]
    obj_vals_to_augment['obj_loc'][:, 0] -= row_shift
    obj_vals_to_augment['obj_loc'][:, 1] -= col_shift

    for index, loc in enumerate(obj_vals_to_augment['obj_loc']):
        if (0 < loc[0] < generator_options.image_dimension[0]
                and 0 < loc[1] < generator_options.image_dimension[1]):
            augmented_image[tuple(loc)] = obj_vals_to_augment[
                                    'obj_vals'][index]
            augmented_label[tuple(loc)] = obj_vals_to_augment[
                                    'label_vals'][index]

    if generator_options.save_obj_det_label:
        rect_points = [r - s for r, s in zip(
            obj_vals_to_augment['rect_points'],
            [row_shift, col_shift, row_shift, col_shift])]

        obj_det_label = [obj_vals_to_augment['obj_name']] + rect_points
        return augmented_image, augmented_label, obj_det_label

    return augmented_image, augmented_label


def make_save_dirs():

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


def get_mask(label):
    colormap = np.asarray([[128, 64, 128], [244, 35, 232], [70, 70, 70],
                               [102, 102, 156], [190, 153, 153], [153, 153, 153],
                               [250, 170, 30], [220, 220, 0], [107, 142, 35],
                               [152, 251, 152], [70, 130, 180], [220, 20, 60],
                               [255, 0, 0], [0, 0, 142], [0, 0, 70],
                               [0, 60, 100], [0, 80, 100],[0, 0, 230],
                               [119, 11, 32], [0, 0, 0]])

    return colormap[np.array(label, dtype=np.uint8)]


def save_data(augmented_image, augmented_label, obj_det_label, index):

    cv2.imwrite(os.path.join(
        generator_options.image_save_path,
        generator_options.name_format %
        (index + generator_options.start_index) + '.jpg'),
        augmented_image)

    cv2.imwrite(os.path.join(
        generator_options.label_save_path,
        generator_options.name_format %
        (index + generator_options.start_index) + '.png'),
        augmented_label)

    if generator_options.save_obj_det_label:
        with open(os.path.join(
                generator_options.label_save_path,
                generator_options.name_format %
                (index + generator_options.start_index) + '.csv'), 'w') as f:

            wr = csv.writer(f, delimiter=',')
            [wr.writerow(l) for l in obj_det_label]
    else:
        obj_det_label = None

    if generator_options.save_label_preview:
        plotter.plot_preview(augmented_image, augmented_label,
                             obj_det_label, index)

    if generator_options.save_mask:
        cv2.imwrite(os.path.join(
            generator_options.mask_save_path,
            generator_options.name_format %
            (index + generator_options.start_index) + '.png'),
            get_mask(augmented_label))


def perform_augmentation():

    """
    This function goes through the augment vector and performs augmentation
    for each augment vector.. The result is saved along with semantic segmentation
    labels and object detection labels (if get_obj_det_label is true)...

    In each augment vector, objects in 'what_objects' is taken and pasted on top
    of the 'background_image' in the augment vector...

    The augmented image and label are then saved. Object detection labels are saved
    in a csv (one for every augment vector) if get_obj_det_label is true...
    """

    make_save_dirs()
    obj_det_label = list()
    background_label = np.ones(tuple(
        generator_options.image_dimension)) * (
        arguments.LABEL_DEF_MATLAB['background'])

    for index, vector in enumerate(tqdm.tqdm(
            generate_augmenter_list.augmenter_list,
            desc='Generating synthetic images')):
        augmented_image = vector['background_image']
        augmented_label = background_label.copy()
        obj_det_label.clear()
        for i in range(vector['num_objects_to_place']):

            if generator_options.save_obj_det_label:
                augmented_image, augmented_label, rect_label = (
                    get_augmented_image(augmented_image,
                                        augmented_label,
                                        object_details.objects[
                                        vector['what_objects'][i]],
                                        vector['locations'][i]))
                obj_det_label.append(rect_label)
            else:
                augmented_image, augmented_label = (
                    get_augmented_image(augmented_image,
                                        augmented_label,
                                        object_details.objects[
                                        vector['what_objects'][i]],
                                        vector['locations'][i]))

        save_data(augmented_image, augmented_label, obj_det_label, index)


if __name__ == '__main__':
    perform_augmentation()