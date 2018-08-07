import matplotlib.pyplot as plt
from data_augmentation.data_analyzer.get_tags_keys import tags_keys
import numpy as np
import operator
import copy


def plot(results, ignore_background=True,
         bar_width=0.3, set_fonts=14, fig_width=10,
         fig_height=7, display_title=False, plot_weight=False):

    any_result = copy.deepcopy(results[tags_keys.data_key][
                                   tags_keys.percentage_key][0])

    if ignore_background:
        any_result.pop('background', None)
    #classes = list(any_result.keys())
    classes = np.array(sorted(any_result.items(),
                       key=operator.itemgetter(1)))[:, 0]
    x = np.arange(0, len(classes))

    cmap = plt.cm.get_cmap('Paired')
    colors = [cmap(0.2), cmap(0.5), cmap(0.7)]
    figure = plt.figure(figsize=(fig_width, fig_height))

    if not plot_weight:
        results[tags_keys.data_key].pop(tags_keys.weight_key, None)

    results[tags_keys.data_key].pop(tags_keys.surface_area_key, None)

    for index, (key, value) in enumerate(
            results[tags_keys.data_key].items()):

        figure.add_subplot(2, 1, index + 1)
        for split, dictionary in enumerate(value):
            if ignore_background:
                dictionary.pop('background', None)

            y = [dictionary[cls] for cls in classes]
            plt.bar(x + split*bar_width, y,
                    width=bar_width, align='center', color=colors[split],
                    zorder=3, label=results[tags_keys.info_key][split])

        plt.xticks([])
        plt.ylabel(key, fontsize=set_fonts, labelpad=30 if key in
                   tags_keys.percentage_key else None)
        plt.grid(zorder=0, axis='y')
        plt.tick_params(axis='both', which='major', labelsize=set_fonts)

        if index == 0:
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                       loc=3, ncol=3,
                       borderaxespad=0., prop={'size': set_fonts})

    plt.xticks(x, classes, rotation=80)
    plt.tight_layout()
    plt.show()


def plot_with_area(results, ignore_background=True,
         bar_width=0.25, set_fonts=14, fig_width=12,
         fig_height=5, display_title=False, plot_weight=False):

    any_result = copy.deepcopy(results[tags_keys.data_key][
                                   tags_keys.surface_area_key][0])

    if ignore_background:
        any_result.pop('background', None)
    # classes = list(any_result.keys())
    classes = np.array(sorted(any_result.items(),
                              key=operator.itemgetter(1)))[:, 0]
    x = np.arange(0, len(classes))

    cmap = plt.cm.get_cmap('Paired')
    labels_list = []
    colors = [cmap(0.2), cmap(0.5), cmap(0.7)]
    figure = plt.figure(figsize=(fig_width, fig_height))

    if not plot_weight:
        results[tags_keys.data_key].pop(tags_keys.weight_key, None)

    dictionary = []
    for index, (key, value) in enumerate(
            results[tags_keys.data_key].items()):
        dictionary.append(value[0])
        labels_list.append(key)

    for index, d in enumerate(dictionary):
        y = [d[cls] for cls in classes]
        y = [val/max(y) for val in y]
        plt.bar(x + (index - 1) * bar_width, y,
                width=bar_width, align='center', color=colors[index],
                zorder=3, label=labels_list[index])

    plt.xticks([])
    plt.ylabel('Normalized values', fontsize=set_fonts)
    plt.grid(zorder=0, axis='y')
    plt.tick_params(axis='both', which='major', labelsize=set_fonts)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=3,
               borderaxespad=0., prop={'size': set_fonts})
    plt.xticks(x, classes, rotation=80)
    plt.tight_layout()
    plt.show()
