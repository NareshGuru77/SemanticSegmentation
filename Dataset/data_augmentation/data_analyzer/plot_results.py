import matplotlib.pyplot as plt
from data_analyzer.get_tags_keys import tags_keys
import numpy as np
import operator
import copy
import cycler
import matplotlib as mpl


def plot(results, ignore_background=True,
         bar_width=0.3, set_fonts=14, fig_width=10,
         fig_height=7, plot_weight=False):

    any_result = copy.deepcopy(results[tags_keys.data_key][
                                   tags_keys.percentage_key][0])

    if ignore_background:
        any_result.pop('background', None)

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


def infer_relation(area_vals, percent_vals, variant, set_fonts, combine):

    x = np.arange(0, len(area_vals) / combine)
    area_vals = np.reshape(area_vals,
                           (int(len(area_vals) / combine), combine))
    area_vals = np.sum(area_vals, axis=1)
    percent_vals = np.reshape(percent_vals,
                              (int(len(percent_vals) / combine), combine))
    percent_vals = np.sum(percent_vals, axis=1)

    plt.bar(x + 0.3, area_vals, width=0.3, zorder=3, label=tags_keys.surface_area_key)
    plt.bar(x, percent_vals, width=0.3,
            align='center', zorder=3, label=tags_keys.percentage_key)
    plt.xticks([])
    plt.grid(zorder=0, axis='y')

    plt.tick_params(axis='both', which='major', labelsize=set_fonts)
    plt.xlabel('Combine every 3 classes' if 'full' in variant else
               'Combine every 2 classes', fontsize=set_fonts)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=3,
               borderaxespad=0., prop={'size': set_fonts})
    plt.tight_layout()


def plot_with_area(results, variant, ignore_background=True,
                   bar_width=0.3, set_fonts=14, fig_width=9,
                   fig_height=6, plot_weight=False):

    num_plots = 5
    colormap = plt.cm.get_cmap('Dark2')
    color = colormap(np.linspace(0, 1, num_plots))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

    any_result = copy.deepcopy(results[tags_keys.data_key][
                                   tags_keys.surface_area_key][0])

    if ignore_background:
        any_result.pop('background', None)

    classes = np.array(sorted(any_result.items(),
                              key=operator.itemgetter(1)))[:, 0]
    x = np.arange(0, len(classes))

    figure = plt.figure(figsize=(fig_width, fig_height))

    if not plot_weight:
        results[tags_keys.data_key].pop(tags_keys.weight_key, None)

    results[tags_keys.data_key].pop(tags_keys.count_key, None)

    area_vals = [results[tags_keys.data_key][tags_keys.surface_area_key][0][cls]
                 for cls in classes]
    area_vals = [val / sum(area_vals) for val in area_vals]

    percent_vals = [results[tags_keys.data_key][tags_keys.percentage_key][0][cls]
                    for cls in classes]
    percent_vals = [val / sum(percent_vals) for val in percent_vals]

    figure.add_subplot(2, 1, 1)
    infer_relation(area_vals, percent_vals, variant, set_fonts,
                   3 if 'full' in variant else 2)

    figure.add_subplot(2, 1, 2)
    plt.bar(x + bar_width, area_vals, width=bar_width, align='center',
            zorder=3, label=tags_keys.surface_area_key)
    plt.bar(x, percent_vals, width=bar_width, align='center',
            zorder=3, label=tags_keys.percentage_key)

    plt.xticks([])
    plt.ylabel('Normalized values', fontsize=set_fonts)
    plt.grid(zorder=0, axis='y')
    plt.tick_params(axis='both', which='major', labelsize=set_fonts)
    plt.xticks(x, classes, rotation=80)
    plt.tight_layout()
    plt.show()
