from data_augmentation import arguments
from data_augmentation import generate_artificial_images
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

generator_options = arguments.GeneratorOptions()


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
            box_value = len(arguments.LABEL_DEF_MATLAB) + (
                arguments.LABEL_DEF_MATLAB[l[0]])
            for i in range(l[1], l[3] + 1):
                if i < generator_options.image_dimension[0]:
                    label[i, l[2]:l[2] + 3] = box_value
                    label[i, l[4] - 3:l[4]] = box_value

            for i in range(l[2], l[4] + 1):
                if i < generator_options.image_dimension[1]:
                    label[l[1]:l[1] + 3, i] = box_value
                    label[l[3] - 3:l[3], i] = box_value
            objects_in_image.append(l[0])

        unique_objects = np.unique(objects_in_image)
        [plt.plot(0, 0, '-', c=generate_artificial_images.colormap[
                                   arguments.LABEL_DEF_MATLAB[obj]] / 255.,
                  label=obj)
         for obj in unique_objects]

        leg = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                         loc=3, ncol=2, mode="expand",
                         borderaxespad=0., prop={'size': 20})

        for handle, line, text in zip(leg.legendHandles,
                                      leg.get_lines(), leg.get_texts()):
            handle.set_linewidth(15)
            text.set_color(line.get_color())

    plt.imshow(generate_artificial_images.colormap[
                   np.array(label, dtype=np.uint8)])

    save_path = os.path.join(
                    generator_options.preview_save_path,
                    generator_options.name_format %
                    (index + generator_options.start_index) + '.png')
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
