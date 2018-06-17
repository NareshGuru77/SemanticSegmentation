from data_augmentation import arguments
from data_augmentation import get_backgrounds_and_data
from data_augmentation import object_details
import numpy as np
import random
from scipy.spatial.distance import cdist

generator_options = arguments.GeneratorOptions()


def remove_clutter(augmenter_list, regenerate_count):
    """
    This function removes vectors with too many objects and too much object occlusion
    """

    removed_vectors = 0
    for index, vector in enumerate(augmenter_list):
        vector_area = 0
        for i in range(vector['num_objects_to_place']):
            vector_area += object_details.objects[vector['what_objects'][i]]['obj_area']
        dist_btw_locations = cdist(vector['locations'], vector['locations'])
        np.fill_diagonal(dist_btw_locations, np.inf)

        if (vector_area > np.product(generator_options.image_dimension)
                * generator_options.max_occupied_area or
                np.any(dist_btw_locations < generator_options.min_distance)):
            del augmenter_list[index]
            removed_vectors += 1

    if (regenerate_count < generator_options.num_regenerate
            and removed_vectors is not 0):
        regenerate_count += 1
        create_augmenter_list(is_regeneration=True,
                              removed_vectors=removed_vectors,
                              regenerate_count=regenerate_count,
                              augmenter_list=augmenter_list)


def get_random_locations(num_objects_to_place):
    """
    Generate a list of random (x,y) points..
    """
    location = [[random.randrange(0, 440, 120), random.randrange(0, 600, 120)]
                for _ in range(num_objects_to_place)]

    return np.array(location)


def create_augmenter_list(is_regeneration=False, removed_vectors=None,
                          regenerate_count=None, augmenter_list=None):

    num_objects = object_details.num_of_objects
    objects_index = np.arange(0, num_objects)

    if is_regeneration:
        augmenter_list = augmenter_list
        num_images = removed_vectors
        regenerate_count = regenerate_count
    else:
        augmenter_list = []
        num_images = generator_options.num_images
        regenerate_count = 0

    for i in range(num_images):
        num_objects_to_place = np.random.randint(1,
                                                 high=generator_options.max_objects)
        what_objects = [objects_index[i] for i in range(num_objects_to_place)]

        background_images = get_backgrounds_and_data.background_images
        if i % len(background_images) == 0:
            np.random.shuffle(background_images)

        np.random.shuffle(objects_index)

        augmenter_list.append({'background_image': background_images[
            i % len(background_images)],
                                    'num_objects_to_place': num_objects_to_place,
                                    'what_objects': what_objects,
                                    'locations': get_random_locations(num_objects_to_place)})

    if generator_options.remove_clutter:
        remove_clutter(augmenter_list,
                       regenerate_count)

    return augmenter_list


augmenter_list = create_augmenter_list()