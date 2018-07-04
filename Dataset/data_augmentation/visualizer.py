from data_augmentation.arguments import generator_options, LABEL_DEF_MATLAB
from data_augmentation.get_backgrounds_and_data import fetch_image_gt_paths
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


colormap = np.asarray([[128, 64, 128], [244, 35, 232], [70, 70, 70],
                       [102, 102, 156], [190, 153, 153], [153, 153, 153],
                       [250, 170, 30], [220, 220, 0], [107, 142, 35],
                       [152, 251, 152], [70, 130, 180], [220, 20, 60],
                       [255, 0, 0], [0, 0, 142], [0, 0, 70],
                       [0, 60, 100], [0, 80, 100], [0, 0, 230],
                       [119, 11, 32], [255, 255, 255], [0, 0, 0],
                       [0, 204, 255], [20, 0, 255], [10, 190, 212],
                       [0, 153, 255], [0, 41, 255], [0, 255, 204],
                       [41, 0, 255], [41, 255, 0], [173, 0, 255],
                       [25, 194, 194], [71, 0, 255], [122, 0, 255],
                       [0, 255, 184], [0, 92, 255], [184, 255, 0],
                       [0, 133, 255], [255, 154, 0]])


def plot_preview(image, label, obj_det_label, index):

    """
    This function can be used to plot a preview image,
    which shows the image and labels alongside each other.

    :param image: Image to plot.
    :param label: Corresponding segmentation label.
    :param obj_det_label: Corresponding object detection label.
                          Can be None.
    :param index: The index number of the image.
    :return: Nothing is returned.
    """

    label = label.copy()

    figure = plt.figure()
    figure.set_figheight(15)
    figure.set_figwidth(20)
    figure.add_subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    figure.add_subplot(1, 2, 2)

    if obj_det_label is not None:
        objects_in_image = []
        for l in obj_det_label:
            box_value = LABEL_DEF_MATLAB[l[0]]
            for i in range(l[1], l[3] + 1):
                if i < generator_options.image_dimension[0]:
                    label[i, l[2]:l[2] + 3] = box_value
                    label[i, l[4] - 3:l[4]] = box_value

            for i in range(l[2], l[4] + 1):
                if i < generator_options.image_dimension[1]:
                    label[l[1]:l[1] + 3, i] = box_value
                    label[l[3] - 3:l[3], i] = box_value
            objects_in_image.append(l[0])

        unique_objects = sorted(np.unique(objects_in_image),
                                key=lambda k: len(k))
        [plt.plot(0, 0, '-', c=colormap[
                                   LABEL_DEF_MATLAB[obj]] / 255.,
                  label=obj)
         for obj in unique_objects]

        leg = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                         loc=3, ncol=2, mode="expand",
                         borderaxespad=0., prop={'size': 20})

        for handle, line, text in zip(leg.legendHandles,
                                      leg.get_lines(), leg.get_texts()):
            handle.set_linewidth(15)
            text.set_color(line.get_color())

    plt.imshow(colormap[
                   np.array(label, dtype=np.uint8)])

    save_path = os.path.join(
                    generator_options.preview_save_path,
                    generator_options.name_format %
                    (index + generator_options.start_index) + '.png')
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def save_overlay(image, label, index):

    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), aspect='auto')
    ax.imshow(label, cmap='Accent', interpolation='none',
              alpha=generator_options.overlay_opacity, aspect='auto')

    save_path = os.path.join(
        generator_options.overlay_save_path,
        generator_options.name_format %
        (index + generator_options.start_index) + '.png')
    plt.xticks([])
    plt.yticks([])
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.axis('off')

    plt.savefig(save_path, bbox_inches="tight", dpi=figure.dpi)
    plt.close(figure)


def save_visuals(image, label, obj_det_label, index):

    if generator_options.save_mask:
        cv2.imwrite(os.path.join(
            generator_options.mask_save_path,
            generator_options.name_format %
            (index + generator_options.start_index) + '.png'),
            colormap[np.array(label, dtype=np.uint8)])

    if generator_options.save_label_preview:
        plot_preview(image, label, obj_det_label, index)

    if generator_options.save_overlay:
        save_overlay(image, label, index)


def get_mask_overlay():

    _, object_files = fetch_image_gt_paths()

    for key in LABEL_DEF_MATLAB:
        if key is not 'background':
            data_list = object_files[key]
            for index, data in enumerate(data_list):
                img = cv2.imread(data[0])
                label = cv2.imread(data[1], 0)
                name = data[1].split('/')[-1].split('.')[0]
                save_visuals(img, label, None, name)
