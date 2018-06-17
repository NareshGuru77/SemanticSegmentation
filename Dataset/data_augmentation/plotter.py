from data_augmentation import arguments
import matplotlib.pyplot as plt
import cv2
import os

generator_options = arguments.GeneratorOptions()


def plot_preview(image, label, obj_det_label, index):

    label = label.copy()
    if obj_det_label is not None:
        for l in obj_det_label:
            for i in range(l[1], l[3] + 1):
                if i < generator_options.image_dimension[0]:
                    label[i, l[2]:l[2] + 3] = len(
                            arguments.LABEL_DEF_MATLAB) + 1
                    label[i, l[4] - 3:l[4]] = len(
                            arguments.LABEL_DEF_MATLAB) + 1

            for i in range(l[2], l[4] + 1):
                if i < generator_options.image_dimension[1]:
                    label[l[1]:l[1] + 3, i] = len(
                            arguments.LABEL_DEF_MATLAB) + 1
                    label[l[3] - 3:l[3], i] = len(
                            arguments.LABEL_DEF_MATLAB) + 1

    figure = plt.figure()
    figure.set_figheight(15)
    figure.set_figwidth(15)
    figure.add_subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    figure.add_subplot(1, 2, 2)
    plt.imshow(label)
    save_path = os.path.join(
                    generator_options.preview_save_path,
                    generator_options.name_format %
                    (index + generator_options.start_index) + '.png')
    plt.savefig(save_path)
    result = cv2.imread(save_path, 1)
    result = result[500:1000, 100:1400, :]
    cv2.imwrite(save_path, result)
    plt.close(figure)
