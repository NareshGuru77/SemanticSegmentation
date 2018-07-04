from data_augmentation.arguments import generator_options
from data_augmentation.generate_artificial_images import perform_augmentation
from data_augmentation.visualizer import save_visuals
from data_augmentation.saver import make_save_dirs
from data_augmentation.get_backgrounds_and_data import fetch_image_gt_paths
import cv2
import csv


def read_files_and_visualize(paths_list):
    """
    This function reads all the images and corresponding
    labels and calls the visualizer.
    :param paths_list: List containing paths to images and labels.
    :return: No returns.
    """
    for index, data in enumerate(paths_list):
        image = cv2.imread(data[0])
        label = cv2.imread(data[1], 0)
        name = data[1].split('/')[-1].split('.')[0]
        obj_label = None

        if generator_options.save_label_preview:
            obj_label = []
            with open(data[2], 'r') as f:
                obj = csv.reader(f, delimiter=',')
                for row in obj:
                    row = [int(r.split('.')[0]) if index != 0 else r
                           for index, r in enumerate(row)]
                    obj_label.append(row)

        save_visuals(image, label, obj_label, name)


if __name__ == '__main__':

    if generator_options.mode == 1:
        perform_augmentation()
    else:
        make_save_dirs()
        data_paths = fetch_image_gt_paths()
        read_files_and_visualize(data_paths)
