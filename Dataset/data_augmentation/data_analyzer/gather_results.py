from data_augmentation.data_analyzer import data_analysis
from data_augmentation.data_analyzer.get_tags_keys import tags_keys


def generate_results(analyzer_train, analyzer_validation,
                     analyzer_test, set_background_weight=None):

    results = {tags_keys.data_key: {tags_keys.percentage_key: [],
                                    tags_keys.count_key: [],
                                    tags_keys.weight_key: []},
               tags_keys.info_key: [tags_keys.training_tag,
                                    tags_keys.validation_tag,
                                    tags_keys.test_tag]}

    percentage_list = [analyzer_train.get_cls_to_percentage(),
                       analyzer_validation.get_cls_to_percentage(),
                       analyzer_test.get_cls_to_percentage()]

    results[tags_keys.data_key][tags_keys.percentage_key] = percentage_list

    count_list = [analyzer_train.get_cls_to_count(percentage_list[0]),
                  analyzer_validation.get_cls_to_count(percentage_list[1]),
                  analyzer_test.get_cls_to_count(percentage_list[2])]

    results[tags_keys.data_key][tags_keys.count_key] = count_list

    weight_list = [analyzer_train.get_cls_to_weight(
        percentage_list[0], set_background_weight=set_background_weight),
                   analyzer_validation.get_cls_to_weight(
        percentage_list[1], set_background_weight=set_background_weight),
                   analyzer_test.get_cls_to_weight(
        percentage_list[2], set_background_weight=set_background_weight)]

    results[tags_keys.data_key][tags_keys.weight_key] = weight_list

    return results