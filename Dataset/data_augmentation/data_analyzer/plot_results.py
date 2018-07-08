import matplotlib.pyplot as plt
from data_augmentation.data_analyzer.get_tags_keys import tags_keys
import numpy as np
import operator
import copy


def plot(results, ignore_background=True,
         bar_width=0.3, set_fonts=14, fig_width=10,
         fig_height=6, display_title=False, plot_weight=False):

    any_result = copy.deepcopy(results[tags_keys.data_key][
                                   tags_keys.percentage_key][0])

    if ignore_background:
        any_result.pop('background', None)
    #classes = list(any_result.keys())
    classes = np.array(sorted(any_result.items(),
                            key=operator.itemgetter(1)))[:,0]
    x = np.arange(0, len(classes))

    cmap = plt.cm.get_cmap('Paired')
    colors = [cmap(0.4), cmap(0.6), cmap(0.8)]
    figure = plt.figure(figsize=(fig_width, fig_height))

    if not plot_weight:
        results[tags_keys.data_key].pop(tags_keys.weight_key, None)

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
        plt.ylabel(key, fontsize=set_fonts)
        plt.grid(zorder=0, axis='y')
        plt.tick_params(axis='both', which='major', labelsize=set_fonts)

        if index == 0:
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                       loc=3, ncol=3,
                       borderaxespad=0., prop={'size': set_fonts})

    plt.xticks(x, classes, rotation=80)
    plt.tight_layout()
    plt.show()